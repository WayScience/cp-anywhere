use cp_anywhere::outline_overlay::{
    render, Background, Brightness, DisplayMode, LineMode, Outline, Precision,
};
use numpy::{PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type OutlineInput<'py> = (PyReadonlyArrayDyn<'py, u32>, [u8; 3]);
type Rendered<'py> = (Vec<usize>, Bound<'py, PyArray1<f32>>);

#[pyfunction]
pub(super) fn render_outline_overlay<'py>(
    py: Python<'py>,
    background: Option<PyReadonlyArrayDyn<'py, f64>>,
    outlines: Vec<OutlineInput<'py>>,
    options: (&str, &str, &str),
    float64: bool,
) -> PyResult<Rendered<'py>> {
    let display = match options.0 {
        "color" => DisplayMode::Color,
        "grayscale" => DisplayMode::Grayscale,
        _ => return Err(PyValueError::new_err("unsupported display mode")),
    };
    let line = match options.1 {
        "inner" => LineMode::Inner,
        "outer" => LineMode::Outer,
        "thick" => LineMode::Thick,
        _ => return Err(PyValueError::new_err("unsupported line mode")),
    };
    let brightness = match options.2 {
        "max_of_image" => Brightness::MaxOfImage,
        "max_possible" => Brightness::MaxPossible,
        _ => return Err(PyValueError::new_err("unsupported brightness")),
    };
    let background = background
        .as_ref()
        .map(|array| {
            Ok::<_, PyErr>(Background {
                shape: array.shape(),
                values: array.as_slice()?,
                precision: if float64 {
                    Precision::Float64
                } else {
                    Precision::Float32
                },
            })
        })
        .transpose()?;
    let outlines = outlines
        .iter()
        .map(|(array, color)| {
            Ok::<_, PyErr>(Outline {
                shape: array.shape(),
                labels: array.as_slice()?,
                color: *color,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let output = py
        .detach(|| render(background, &outlines, display, line, brightness))
        .map_err(PyValueError::new_err)?;
    let values = output
        .values
        .into_iter()
        .map(|value| value as f32)
        .collect();
    Ok((output.shape, PyArray1::from_vec(py, values)))
}
