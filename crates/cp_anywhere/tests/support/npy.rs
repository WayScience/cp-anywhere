//! Minimal NPY v1.0 reader for integration-test fixtures only
//! (little-endian f4/f8/u4/u1, i4, C order). Not part of the shipped
//! library — scoped to the exact format produced by a plain NumPy
//! `array.astype('<f4').tofile`/`np.save` call, rather than adding a
//! general-purpose NPY dependency to the core crate. Mirrors
//! `benches/support/npy.rs` with an added `read_f32`.

use std::fs;
use std::path::Path;

pub struct Npy {
    pub shape: Vec<usize>,
    pub descr: String,
    pub data: Vec<u8>,
}

pub fn read(path: &Path) -> Npy {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "not an NPY file: {path:?}");
    let major = bytes[6];
    let header_len_offset = 8;
    let (header_len, header_start) = if major == 1 {
        let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (len, header_len_offset + 2)
    } else {
        let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (len, header_len_offset + 4)
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len]).unwrap();
    let shape_start = header.find('(').unwrap() + 1;
    let shape_end = header.find(')').unwrap();
    let shape: Vec<usize> = header[shape_start..shape_end]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();
    let descr_start = header.find("descr': '").unwrap() + "descr': '".len();
    let descr_end = header[descr_start..].find('\'').unwrap() + descr_start;
    let descr = header[descr_start..descr_end].to_string();
    let data = bytes[header_start + header_len..].to_vec();
    Npy { shape, descr, data }
}

// Each test binary includes this module via `#[path]` and typically
// uses only a subset of the readers, so silence per-binary dead_code.
#[allow(dead_code)]
pub fn read_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let npy = read(path);
    assert_eq!(npy.descr, "<f4", "{path:?} is not float32: {}", npy.descr);
    let values = npy
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (npy.shape, values)
}

#[allow(dead_code)]
pub fn read_u32(path: &Path) -> (Vec<usize>, Vec<u32>) {
    let npy = read(path);
    assert_eq!(npy.descr, "<i4", "{path:?} is not int32: {}", npy.descr);
    let values = npy
        .data
        .chunks_exact(4)
        .map(|c| {
            let signed = i32::from_le_bytes(c.try_into().unwrap());
            u32::try_from(signed).unwrap_or_else(|_| panic!("negative label in {path:?}"))
        })
        .collect();
    (npy.shape, values)
}
