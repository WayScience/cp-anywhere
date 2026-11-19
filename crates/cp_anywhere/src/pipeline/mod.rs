//! Execute the supported CellProfiler `.cppipe` subset on named arrays.
//! The host supplies decoded image sets and handles output destinations.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

use crate::segmentation::{secondary::SecondaryObjectsResult, SegmentationResult};
use crate::ResultTable;

mod execute;
mod measure;
mod parse;
mod wire;

pub use wire::{unpack_images, ImageDescriptor};

/// Original type of a decoded plane, before CellProfiler pixel conversion.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
}

impl Dtype {
    fn integer_range(self) -> Option<(f64, f64)> {
        match self {
            Self::Uint8 => Some((0.0, u8::MAX as f64)),
            Self::Uint16 => Some((0.0, u16::MAX as f64)),
            Self::Uint32 => Some((0.0, u32::MAX as f64)),
            Self::Uint64 => Some((0.0, u64::MAX as f64)),
            Self::Int8 => Some((i8::MIN as f64, i8::MAX as f64)),
            Self::Int16 => Some((i16::MIN as f64, i16::MAX as f64)),
            Self::Int32 => Some((i32::MIN as f64, i32::MAX as f64)),
            Self::Int64 => Some((i64::MIN as f64, i64::MAX as f64)),
            _ => None,
        }
    }
}

/// Row-major decoded image or caller-supplied label plane.
#[derive(Debug)]
pub struct Plane {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
    pub dtype: Dtype,
}

impl Plane {
    fn validate(&self) -> Result<(), String> {
        if pixel_count(&self.shape)? != self.values.len() {
            return Err("pixel buffer length must match dimensions".into());
        }
        if let Some((min, max)) = self.dtype.integer_range() {
            // The 64-bit maxima round up to the first out-of-range f64.
            let upper_exclusive = matches!(self.dtype, Dtype::Uint64 | Dtype::Int64);
            if self
                .values
                .iter()
                .any(|x| x.fract() != 0.0 || *x < min || *x > max || (upper_exclusive && *x == max))
            {
                return Err(
                    "pixels must be exactly representable integers in the declared dtype".into(),
                );
            }
        } else if matches!(self.dtype, Dtype::Bool)
            && self.values.iter().any(|x| !matches!(*x, 0.0 | 1.0))
        {
            return Err("bool pixels must be zero or one".into());
        }
        Ok(())
    }

    pub(crate) fn converted(&self) -> Vec<f64> {
        let range = self.dtype.integer_range();
        self.values
            .iter()
            .map(|&pixel| {
                let pixel = pixel as f32;
                match range {
                    Some((min, max)) if min < 0.0 => {
                        ((pixel - min as f32) / (max - min) as f32).clamp(0.0, 1.0) as f64
                    }
                    Some((_, max)) => (pixel / max as f32) as f64,
                    None => pixel as f64,
                }
            })
            .collect()
    }

    pub(crate) fn labels(&self) -> Result<Vec<u32>, String> {
        if self.dtype.integer_range().is_none() {
            return Err("label arrays must contain nonnegative integers".into());
        }
        self.values
            .iter()
            .map(|&x| {
                if (0.0..=u32::MAX as f64).contains(&x) && x.fract() == 0.0 {
                    Ok(x as u32)
                } else {
                    Err("label arrays must contain nonnegative integers that fit in uint32".into())
                }
            })
            .collect()
    }
}

fn pixel_count(shape: &[usize]) -> Result<usize, String> {
    if !matches!(shape.len(), 2 | 3) || shape.contains(&0) {
        return Err("dimensions must describe a nonempty 2D or 3D array".into());
    }
    shape.iter().try_fold(1usize, |total, &n| {
        total
            .checked_mul(n)
            .ok_or_else(|| "image dimensions overflow".into())
    })
}

#[derive(Debug)]
pub struct TertiaryObjectsResult {
    pub labels: Vec<u32>,
}

/// Named segmentation outputs, corrected images, and optional measurements.
#[derive(Debug)]
pub struct RunResult {
    pub primary: BTreeMap<String, SegmentationResult>,
    pub secondary: BTreeMap<String, SecondaryObjectsResult>,
    pub tertiary: BTreeMap<String, TertiaryObjectsResult>,
    pub derived_images: BTreeMap<String, Plane>,
    pub object_shapes: BTreeMap<String, Vec<usize>>,
    pub measurements: Option<ResultTable>,
}

pub(crate) struct State {
    pub(crate) images: Vec<(String, Plane)>,
    pub(crate) primary: BTreeMap<String, SegmentationResult>,
    pub(crate) secondary: BTreeMap<String, SecondaryObjectsResult>,
    pub(crate) tertiary: BTreeMap<String, TertiaryObjectsResult>,
    raw_labels: BTreeMap<String, Vec<u32>>,
    shapes: BTreeMap<String, Vec<usize>>,
    primary_shapes: BTreeMap<String, Vec<usize>>,
    derived: std::collections::BTreeSet<String>,
}

impl State {
    pub(crate) fn image(&self, name: &str) -> Option<&Plane> {
        self.images
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, plane)| plane)
    }

    pub(crate) fn shape_of(&self, name: &str) -> Option<&[usize]> {
        self.shapes.get(name).map(Vec::as_slice)
    }

    pub(crate) fn objects(&self, name: &str) -> Option<(&[usize], &[u32])> {
        let shape = self.shape_of(name)?;
        let values = self
            .tertiary
            .get(name)
            .map(|r| &r.labels)
            .or_else(|| self.secondary.get(name).map(|r| &r.labels))
            .or_else(|| self.primary.get(name).map(|r| &r.labels))
            .or_else(|| self.raw_labels.get(name))?;
        Some((shape, values))
    }
}

/// A parsed pipeline. Compile once and run once per complete image set.
#[derive(Debug)]
pub struct Pipeline {
    plan: Value,
}

impl Pipeline {
    /// Parse original CellProfiler text without Python or filesystem access.
    pub fn parse(text: &str) -> Result<Self, String> {
        Ok(Self {
            plan: parse::parse(text)?,
        })
    }

    /// Inspect translated steps, skipped modules, and explicit save/export settings.
    pub fn plan(&self) -> &Value {
        &self.plan
    }

    /// Run the numerical stages without image or database side effects.
    pub fn run(&self, images: Vec<(String, Plane)>) -> Result<RunResult, String> {
        execute::run(&self.plan, images)
    }
}
