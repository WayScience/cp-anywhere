//! Standalone CellProfiler 4.2.8 IdentifyPrimaryObjects declumping.
//! No Python, filesystem, or external-crate dependency.

/// Legacy `numpy.random.RandomState` MT19937 generator, matching
/// `numpy.random.seed(0); numpy.random.uniform(0, 0.001, shape)` bit
/// for bit. CellProfiler's Shape unclump method reseeds the GLOBAL
/// legacy numpy RNG with a fixed seed (0) before jittering the
/// distance transform (`numpy.random.seed(0); ... += numpy.random.uniform(...)`),
/// so a from-scratch, standalone generator is the only way to
/// reproduce that jitter exactly (no external crate dependency is
/// allowed in this crate). Only the plain-integer-seed path is
/// implemented (`mt19937_seed`'s Knuth scalar seeding) -- the
/// alternate `init_by_array` seeding path is for array/SeedSequence
/// seeds, which CellProfiler's own `numpy.random.seed(0)` call never
/// takes.
struct Mt19937 {
    key: [u32; 624],
    pos: usize,
}
impl Mt19937 {
    fn seed(seed: u32) -> Self {
        let mut key = [0u32; 624];
        let mut s = seed;
        for (pos, slot) in key.iter_mut().enumerate() {
            *slot = s;
            s = 1812433253u32
                .wrapping_mul(s ^ (s >> 30))
                .wrapping_add(pos as u32 + 1);
        }
        Mt19937 { key, pos: 624 }
    }
    fn gen(&mut self) {
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908b0df;
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;
        for i in 0..624 {
            let y = (self.key[i] & UPPER_MASK) | (self.key[(i + 1) % 624] & LOWER_MASK);
            let mag = if y & 1 != 0 { MATRIX_A } else { 0 };
            self.key[i] = self.key[(i + M) % 624] ^ (y >> 1) ^ mag;
        }
        self.pos = 0;
    }
    fn next_u32(&mut self) -> u32 {
        if self.pos == 624 {
            self.gen();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
    fn next_double(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }
    /// `numpy.random.uniform(low, high, n)`.
    fn uniform(&mut self, low: f64, high: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|_| low + (high - low) * self.next_double())
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnclumpMethod {
    None,
    Intensity,
    Shape,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WatershedMethod {
    None,
    Intensity,
    Shape,
    Propagate,
}
#[derive(Clone, Copy, Debug)]
pub enum Smoothing {
    Automatic,
    Manual(f64),
}
#[derive(Clone, Copy, Debug)]
pub enum MaximaSuppression {
    Automatic,
    Manual(f64),
}
#[derive(Clone, Debug)]
pub struct DeclumpConfig {
    pub unclump: UnclumpMethod,
    pub watershed: WatershedMethod,
    pub smoothing: Smoothing,
    pub suppression: MaximaSuppression,
    pub minimum_diameter: f64,
    pub low_resolution_maxima: bool,
    pub basic: bool,
    pub fill_holes_for_shape: bool,
}
impl Default for DeclumpConfig {
    fn default() -> Self {
        Self {
            unclump: UnclumpMethod::Intensity,
            watershed: WatershedMethod::Intensity,
            smoothing: Smoothing::Automatic,
            suppression: MaximaSuppression::Automatic,
            minimum_diameter: 10.0,
            low_resolution_maxima: true,
            basic: false,
            fill_holes_for_shape: false,
        }
    }
}
#[derive(Clone, Debug)]
pub struct DeclumpResult {
    pub labels: Vec<u32>,
    // Not yet consumed by this crate's own `identify_primary_objects`
    // caller (which only needs `.labels`), but kept as real public
    // API for future diagnostic exposure (mirroring `threshold_map`/
    // `guide_threshold` on `SegmentationResult`), not dead weight.
    #[allow(dead_code)]
    pub object_count: u32,
    #[allow(dead_code)]
    pub labeled_maxima: Vec<u32>,
    #[allow(dead_code)]
    pub maxima_suppression_size: f64,
}
/// Arrays are row-major, `width * height` pixels. Labels are the existing
/// foreground connected components; zero is background. `object_count` is
/// preserved on the no-op path, just as CellProfiler does.
pub fn declump(
    image: &[f64],
    mask: Option<&[bool]>,
    labels: &[u32],
    width: usize,
    height: usize,
    object_count: u32,
    config: &DeclumpConfig,
) -> Result<DeclumpResult, String> {
    let n = width
        .checked_mul(height)
        .ok_or("image dimensions overflow")?;
    if n != image.len() || n != labels.len() || mask.is_some_and(|m| m.len() != n) {
        return Err("image, labels and mask must match width * height".into());
    }
    if !config.basic
        && (config.unclump == UnclumpMethod::None || config.watershed == WatershedMethod::None)
    {
        return Ok(DeclumpResult {
            labels: labels.to_vec(),
            object_count,
            labeled_maxima: vec![0; n],
            maxima_suppression_size: 7.0,
        });
    }
    let owned_mask;
    let mask = if let Some(m) = mask {
        m
    } else {
        owned_mask = vec![true; n];
        &owned_mask
    };
    let smoothing = match config.smoothing {
        Smoothing::Automatic => 2.35 * config.minimum_diameter / 3.5,
        Smoothing::Manual(x) => x,
    };
    let blurred = smooth(image, mask, width, height, smoothing);
    let resize = if config.minimum_diameter > 10.0 && (config.basic || config.low_resolution_maxima)
    {
        10.0 / config.minimum_diameter
    } else {
        1.0
    };
    let suppression = if resize < 1.0 {
        if config.basic || matches!(config.suppression, MaximaSuppression::Automatic) {
            7.0
        } else {
            match config.suppression {
                MaximaSuppression::Manual(x) => x * resize + 0.5,
                _ => unreachable!(),
            }
        }
    } else {
        if config.basic {
            config.minimum_diameter / 1.5
        } else {
            match config.suppression {
                MaximaSuppression::Automatic => config.minimum_diameter / 1.5,
                MaximaSuppression::Manual(x) => x,
            }
        }
    };
    if resize < 1.0 {
        return Err("low resolution not yet implemented".into());
    }
    // Shape unclump: distance-transform the (hole-filled, if the main
    // fill_holes setting is "Never") foreground, jitter it with the
    // SAME fixed-seed(0) legacy RNG draw CellProfiler itself uses (to
    // break exact distance ties into unique maxima), then find maxima
    // on THIS jittered array instead of the blurred intensity image.
    let mut shape_distance: Option<Vec<f64>> = None;
    let maxima_source: Vec<f64> = if !config.basic && config.unclump == UnclumpMethod::Shape {
        let mut foreground: Vec<bool> = labels.iter().map(|&l| l > 0).collect();
        if config.fill_holes_for_shape {
            let mut filled = labels.to_vec();
            super::holes::fill_labeled_holes(&mut filled, height, width, None);
            foreground = filled.iter().map(|&l| l > 0).collect();
        }
        let mut d = distance_transform_edt(&foreground, width, height);
        let jitter = Mt19937::seed(0).uniform(0.0, 0.001, n);
        for (v, j) in d.iter_mut().zip(&jitter) {
            *v += j;
        }
        shape_distance = Some(d.clone());
        d
    } else {
        blurred
    };
    let binary = maxima(
        &maxima_source,
        labels,
        width,
        height,
        (suppression - 0.5).max(1.0),
    );
    let (labeled_maxima, count) = label8(&binary, width, height);
    let output = if !config.basic && config.watershed == WatershedMethod::Propagate {
        propagate(labels, &labeled_maxima, width, height)
    } else if !config.basic && config.watershed == WatershedMethod::Shape {
        // Reuses the SAME (already-jittered) distance transform
        // computed above when unclump was also Shape -- matching
        // CellProfiler's own `if distance_transformed_image is None`
        // check, which is false in that case. Only recomputes a
        // fresh, unjittered one when Shape watershed is paired with a
        // different unclump method.
        let d = match shape_distance {
            Some(d) => d,
            None => {
                let foreground: Vec<bool> = labels.iter().map(|&l| l > 0).collect();
                distance_transform_edt(&foreground, width, height)
            }
        };
        let max = d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Real CellProfiler: watershed_image = -d; watershed_image -=
        // min(watershed_image). Since min(-d) = -max(d), this
        // simplifies to max(d) - d (NOT -d - min(d)).
        let elevation: Vec<f64> = d.iter().map(|&v| max - v).collect();
        watershed(&elevation, labels, &labeled_maxima, width, height)
    } else {
        let elevation = image.iter().map(|&x| 1.0 - x).collect::<Vec<_>>();
        watershed(&elevation, labels, &labeled_maxima, width, height)
    };
    Ok(DeclumpResult {
        labels: output,
        object_count: count,
        labeled_maxima,
        maxima_suppression_size: suppression / resize,
    })
}
fn smooth(image: &[f64], mask: &[bool], w: usize, h: usize, size: f64) -> Vec<f64> {
    if size == 0.0 {
        return image.to_vec();
    }
    let sigma = size / 2.35;
    let radius = ((size / 2.0) as usize).max(1);
    let mut weights = Vec::with_capacity(radius + 1);
    for r in 0..=radius {
        weights.push(
            1.0 / (2.0 * std::f64::consts::PI).sqrt() / sigma
                * (-0.5 * (r as f64).powi(2) / sigma.powi(2)).exp(),
        );
    }
    let convolve = |src: &[f64], axis: usize| {
        let mut dst = vec![0.0; src.len()];
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                let mut v = src[p] * weights[0];
                for r in (1..=radius).rev() {
                    let (a, b) = if axis == 0 {
                        (
                            if y >= r { src[(y - r) * w + x] } else { 0.0 },
                            if y + r < h { src[(y + r) * w + x] } else { 0.0 },
                        )
                    } else {
                        (
                            if x >= r { src[p - r] } else { 0.0 },
                            if x + r < w { src[p + r] } else { 0.0 },
                        )
                    };
                    v += (a + b) * weights[r];
                }
                dst[p] = v;
            }
        }
        dst
    };
    let m = mask
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let masked = image
        .iter()
        .zip(mask)
        .map(|(&x, &b)| if b { x } else { 0.0 })
        .collect::<Vec<_>>();
    let edge = convolve(&convolve(&m, 0), 1);
    let blurred = convolve(&convolve(&masked, 0), 1);
    blurred
        .iter()
        .zip(edge)
        .zip(mask)
        .map(|((&x, e), &b)| if b { x / e } else { 0.0 })
        .collect()
}

// Packed, bit-indexed binary_shrink lookup tables from centrosome 1.2.3
// cpmorphology.binary_shrink, in its UL/R, UR/B, LR/L, LL/T order.
const SHRINK_TABLES: [[u64; 8]; 4] = [
    [
        0xffffffff7475ffff,
        0xffffffff74ffffff,
        0xffffffffffffffff,
        0xffffffff74ffffff,
        0xffffffffffffffff,
        0xffffffffffffffff,
        0xffffffffffffffff,
        0xffffffff74ffffff,
    ],
    [
        0x7f2fffff7723ffff,
        0x7fffffff77ffffff,
        0xffffffffffffffff,
        0xffffffffffffffff,
        0x7f2fffffffffffff,
        0x7fffffffffffffff,
        0xffffffffffffffff,
        0xffffffffffffffff,
    ],
    [
        0xff2effffffffffff,
        0xffffffffffffffff,
        0xffffffffffffffff,
        0xffffffffffffffff,
        0xff2efffffffeffff,
        0xffffffffffffffff,
        0xff2efffffffeffff,
        0xff2efffffffeffff,
    ],
    [
        0xffffffffffffffff,
        0xfffffffffcfeffff,
        0xfffffffffffeffff,
        0xfffffffffcfeffff,
        0xffffffffffffffff,
        0xffffffffffffffff,
        0xffeefffffffeffff,
        0xcceefffffcfeffff,
    ],
];
fn shrink(mut pixels: Vec<bool>, w: usize, h: usize) -> Vec<bool> {
    let mut active = pixels
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| b.then_some(i))
        .collect::<Vec<_>>();
    loop {
        let count = active.len();
        for table in SHRINK_TABLES {
            let mut remove = Vec::new();
            for &p in &active {
                let mut index = 0usize;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if let Some(q) = neighbor(p, w, h, dy, dx) {
                            if pixels[q] {
                                index |= 1 << ((dy + 1) * 3 + dx + 1);
                            }
                        }
                    }
                }
                if table[index / 64] & (1 << (index % 64)) == 0 {
                    remove.push(p);
                }
            }
            for p in remove {
                pixels[p] = false;
            }
            active.retain(|&p| pixels[p]);
        }
        if active.len() == count {
            break;
        }
    }
    pixels
}
pub(crate) fn neighbor(p: usize, w: usize, h: usize, dy: isize, dx: isize) -> Option<usize> {
    let y = (p / w).checked_add_signed(dy)?;
    let x = (p % w).checked_add_signed(dx)?;
    (y < h && x < w).then_some(y * w + x)
}
fn label8(binary: &[bool], w: usize, h: usize) -> (Vec<u32>, u32) {
    let mut out = vec![0; binary.len()];
    let mut count = 0;
    let mut todo = Vec::new();
    for p in 0..binary.len() {
        if !binary[p] || out[p] != 0 {
            continue;
        }
        count += 1;
        out[p] = count;
        todo.push(p);
        while let Some(q) = todo.pop() {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(r) = neighbor(q, w, h, dy, dx) {
                        if binary[r] && out[r] == 0 {
                            out[r] = count;
                            todo.push(r);
                        }
                    }
                }
            }
        }
    }
    (out, count)
}

/// 1D exact squared-distance transform of a sampled function (the
/// lower envelope of unit parabolas), Felzenszwalt & Huttenlocher
/// 2004 -- the same algorithm `scipy.ndimage.distance_transform_edt`
/// itself uses internally, so results match to float precision (this
/// measures a well-defined mathematical quantity, not a heuristic:
/// any correct implementation agrees).
#[allow(clippy::needless_range_loop)]
fn edt_1d(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    let mut k = 0usize;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let mut s = ((f[q] + (q * q) as f64) - (f[v[k]] + (v[k] * v[k]) as f64))
            / (2.0 * q as f64 - 2.0 * v[k] as f64);
        while k > 0 && s <= z[k] {
            k -= 1;
            s = ((f[q] + (q * q) as f64) - (f[v[k]] + (v[k] * v[k]) as f64))
                / (2.0 * q as f64 - 2.0 * v[k] as f64);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        if k + 1 < z.len() {
            z[k + 1] = f64::INFINITY;
        }
    }
    let mut d = vec![0.0f64; n];
    let mut k = 0usize;
    for q in 0..n {
        while k < n && z[k + 1] < q as f64 {
            k += 1;
        }
        let dv = q as f64 - v[k] as f64;
        d[q] = dv * dv + f[v[k]];
    }
    d
}

/// `scipy.ndimage.distance_transform_edt`: for each foreground pixel,
/// the Euclidean distance to the nearest background (false) pixel;
/// background pixels are 0. Separable: a vertical pass then a
/// horizontal pass of 1D squared-distance transforms, final sqrt.
pub(crate) fn distance_transform_edt(foreground: &[bool], w: usize, h: usize) -> Vec<f64> {
    // Degenerate no-background case: real scipy 1.9.3 (the pinned
    // CellProfiler 4.2.8 environment's scipy.ndimage) does NOT return
    // a constant for an image with no background pixel at all. Its
    // feature transform (`NI_EuclideanFeatureTransform`) zero-
    // initializes the feature array (scipy/ndimage/_morphology.py,
    // `ft = numpy.zeros((input.ndim,) + input.shape, ...)`), its base
    // pass writes only `features[0] = -1` for foreground pixels
    // (scipy/ndimage/src/ni_morphology.c, `_ComputeFT`), and a
    // `_VoronoiFT` line with no background site leaves every feature
    // untouched, so every pixel ends up measuring against the stale
    // (-1, 0) feature: d[row][col] = sqrt((row + 1)^2 + col^2), a
    // strictly monotone bowl rising toward the bottom-right corner.
    // CellProfiler's Shape unclump depends on this quirk: on an
    // all-foreground thresholded image (the all-zero stress field), the
    // seed-0 jitter is far too small to dent the bowl's ~1-pixel
    // gradient, leaving exactly ONE local maximum, so real CP declumps
    // the whole image into ONE object that the max-size filter then
    // erases. The sentinel-based path below would instead return a
    // constant distance, whose jitter fabricates thousands of spurious
    // maxima (verified against the real scipy function; see the
    // all-foreground regression tests).
    if foreground.iter().all(|&f| f) {
        return (0..h * w)
            .map(|i| {
                let row = (i / w + 1) as f64;
                let col = (i % w) as f64;
                (row * row + col * col).sqrt()
            })
            .collect();
    }
    // A large *finite* sentinel, not f64::INFINITY: the parabola
    // intersection formula subtracts two f-values, and inf - inf is
    // NaN, which silently breaks the lower-envelope comparisons.
    // Scaled to the image so it is always many orders of magnitude
    // bigger than any real squared coordinate (keeping the lower-
    // envelope comparisons correct) while staying small enough that
    // f64's ~15-16 significant decimal digits still resolve integer
    // q^2 - v^2 differences precisely (a fixed 1e18 constant was
    // tried first and silently reintroduced the same inf-inf-style
    // precision collapse this sentinel is meant to avoid).
    let inf = ((w * w + h * h) as f64) * 1e6 + 1e6;
    let mut g = vec![0.0f64; foreground.len()];
    for x in 0..w {
        let col: Vec<f64> = (0..h)
            .map(|y| if foreground[y * w + x] { inf } else { 0.0 })
            .collect();
        let d = edt_1d(&col);
        for y in 0..h {
            g[y * w + x] = d[y];
        }
    }
    let mut out = vec![0.0f64; foreground.len()];
    for y in 0..h {
        let row = &g[y * w..y * w + w];
        let d = edt_1d(row);
        for x in 0..w {
            out[y * w + x] = d[x].sqrt();
        }
    }
    out
}

fn maxima(image: &[f64], labels: &[u32], w: usize, h: usize, radius: f64) -> Vec<bool> {
    let r = radius as isize;
    let mut offsets = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if (dy * dy + dx * dx) as f64 <= radius * radius {
                offsets.push((dy, dx));
            }
        }
    }
    let binary = (0..image.len())
        .map(|p| {
            labels[p] > 0
                && image[p] > 0.0
                && offsets.iter().all(|&(dy, dx)| {
                    neighbor(p, w, h, dy, dx)
                        .is_none_or(|q| labels[q] != labels[p] || image[p] >= image[q])
                })
        })
        .collect::<Vec<_>>();
    shrink(binary, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coords(binary: &[bool], w: usize) -> Vec<(usize, usize)> {
        binary
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| b.then_some((i / w, i % w)))
            .collect()
    }

    #[test]
    fn maxima_matches_real_centrosome_on_a_random_image_with_a_large_disk() {
        // Fixed numpy.random.default_rng(42) 12x12 image, top 2 rows
        // masked out of the label (labels=0), suppression disk radius
        // 6.5 (strel_disk(max(1, 7 - 0.5))). Verified against real
        // centrosome.cpmorphology.strel_disk/is_local_maximum/
        // binary_shrink: single surviving maximum at (row=8, col=7).
        #[rustfmt::skip]
        let image: Vec<f64> = vec![
            0.7739560485559633, 0.4388784397520523, 0.8585979199113825, 0.6973680290593639, 0.09417734788764953, 0.9756223516367559, 0.761139701990353, 0.7860643052769538, 0.12811363267554587, 0.45038593789556713, 0.37079802423258124, 0.9267649888486018,
            0.6438651200806645, 0.82276161327083, 0.44341419882733113, 0.2272387217847769, 0.5545847870158348, 0.06381725610417532, 0.8276311719925821, 0.6316643991220648, 0.7580877400853738, 0.35452596812986836, 0.9706980243949033, 0.8931211213221977,
            0.7783834970737619, 0.19463870785196757, 0.4667210037270342, 0.04380376578722878, 0.15428949206754783, 0.6830489532424546, 0.7447621559078171, 0.96750973243421, 0.32582535813815194, 0.3704597060348689, 0.4695558112758079, 0.1894713590842857,
            0.12992150533547164, 0.47570492622593374, 0.2269093490508841, 0.6698139946825103, 0.43715191887233074, 0.8326781960578374, 0.7002651020022491, 0.31236664138204107, 0.8322598013952011, 0.8047643574968019, 0.38747837903017446, 0.2883281039302441,
            0.6824955039749755, 0.1397524836093098, 0.19990820247510832, 0.007362269751005512, 0.7869243775021384, 0.6648508565920321, 0.7051653786263351, 0.7807290310219679, 0.45891577553833995, 0.5687411959528937, 0.13979699812765745, 0.11453007353597344,
            0.6684029617904717, 0.4710962061431325, 0.5652361064811888, 0.7649988574160256, 0.6347183200005908, 0.5535794006579958, 0.5592071607454135, 0.3039500980626122, 0.030817834567939406, 0.43671738923236236, 0.2145846728195292, 0.40852864372463615,
            0.8534030732681661, 0.23393948586534075, 0.05830274168906602, 0.28138389202199654, 0.2935937577666836, 0.6619165147268951, 0.5570321523412783, 0.7838982091064135, 0.6643135403273875, 0.4063868614400705, 0.8140203846660347, 0.1669729199077039,
            0.022712073133860478, 0.09004786077564175, 0.7223593505964503, 0.4618772302513874, 0.1612717790336018, 0.5010447751033635, 0.15231210271316842, 0.696320375077736, 0.4461562755740307, 0.3810212260964825, 0.3015120891478765, 0.6302825931188885,
            0.3618126105533904, 0.087649919316101, 0.11800590212051532, 0.9618976645495145, 0.908580690707607, 0.6997071338107496, 0.2658699614595196, 0.9691763773477239, 0.7787509039657946, 0.7168901891589956, 0.44936150214378867, 0.272241561845159,
            0.0963909621534993, 0.9026023965438417, 0.45577628983361107, 0.20236336479523032, 0.3059566241506525, 0.579219568941896, 0.1767727829392317, 0.8566142840923755, 0.7585195298352101, 0.7194629559509368, 0.4320930397751037, 0.6273088407024432,
            0.5840979689127356, 0.64984660155482, 0.08444432113988909, 0.41580740217060963, 0.04161417386189248, 0.49399081924451893, 0.3298612123327853, 0.1445241888660469, 0.10340296772255164, 0.587644572177712, 0.1705929685368861, 0.9251201183767972,
            0.581061139700395, 0.34686980453483707, 0.5909154914814168, 0.022803871029697498, 0.9585592132414453, 0.48230343694290023, 0.7827352272502862, 0.08272999992243857, 0.48665833083816035, 0.4907069943545209, 0.9378264549749828, 0.5717280523760754,
        ];
        let labels: Vec<u32> = (0..144).map(|p| if p / 12 < 2 { 0 } else { 1 }).collect();
        let binary = maxima(&image, &labels, 12, 12, (7.0_f64 - 0.5).max(1.0));
        assert_eq!(coords(&binary, 12), vec![(8, 7)]);
    }

    #[test]
    fn maxima_matches_real_centrosome_on_a_single_symmetric_peak() {
        // Fixed 5x5 symmetric peak, disk radius 1. Verified against
        // real centrosome: single maximum at (2, 2).
        #[rustfmt::skip]
        let image: Vec<f64> = vec![
            0.1, 0.2, 0.3, 0.2, 0.1,
            0.2, 0.5, 0.9, 0.5, 0.2,
            0.3, 0.9, 1.0, 0.9, 0.3,
            0.2, 0.5, 0.9, 0.5, 0.2,
            0.1, 0.2, 0.3, 0.2, 0.1,
        ];
        let labels = vec![1u32; 25];
        let binary = maxima(&image, &labels, 5, 5, 1.0);
        assert_eq!(coords(&binary, 5), vec![(2, 2)]);
    }

    #[test]
    fn maxima_shrinks_a_flat_plateau_blob_to_a_single_point() {
        // Fixed 5x6 flat plateau peak, disk radius 1. Verified against
        // real centrosome: the blob of touching maxima (including flat
        // border/corner pixels, which legitimately qualify since
        // is_local_maximum's comparison is non-strict) shrinks to a
        // single surviving point at (2, 2).
        #[rustfmt::skip]
        let image: Vec<f64> = vec![
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
            0.1, 0.9, 0.9, 0.9, 0.9, 0.1,
            0.1, 0.9, 0.9, 0.9, 0.9, 0.1,
            0.1, 0.9, 0.9, 0.9, 0.9, 0.1,
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        ];
        let labels = vec![1u32; 30];
        let binary = maxima(&image, &labels, 6, 5, 1.0);
        assert_eq!(coords(&binary, 6), vec![(2, 2)]);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn distance_transform_edt_matches_real_scipy_ndimage() {
        // 8x8 ring with a hole at (3,3), verified against real
        // scipy.ndimage.distance_transform_edt.
        #[rustfmt::skip]
        let mut foreground = vec![false; 64];
        for r in 1..7 {
            for c in 1..7 {
                foreground[r * 8 + c] = true;
            }
        }
        foreground[3 * 8 + 3] = false;
        let d = distance_transform_edt(&foreground, 8, 8);
        #[rustfmt::skip]
        let expected: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.4142135623730951, 1.0, 1.4142135623730951, 2.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0,
            0.0, 1.0, 1.4142135623730951, 1.0, 1.4142135623730951, 2.0, 1.0, 0.0,
            0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        for (got, want) in d.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn distance_transform_edt_matches_real_scipy_on_an_irregular_random_mask() {
        // Fixed numpy.random.default_rng(42) 8x8 boolean mask
        // (rng.random((8,8)) > 0.3), verified against real
        // scipy.ndimage.distance_transform_edt.
        #[rustfmt::skip]
        let foreground: Vec<bool> = vec![
            true, true, true, true, false, true, true, true,
            false, true, true, true, true, true, true, false,
            true, false, true, true, true, true, true, true,
            true, false, true, false, false, true, true, true,
            true, true, true, false, false, true, false, true,
            true, true, true, true, true, true, true, false,
            true, false, false, false, true, true, true, true,
            true, true, false, false, true, true, true, true,
        ];
        let d = distance_transform_edt(&foreground, 8, 8);
        #[rustfmt::skip]
        let expected: Vec<f64> = vec![
            1.0, 1.4142135623730951, 2.0, 1.0, 0.0, 1.0, 1.4142135623730951, 1.0,
            0.0, 1.0, 1.4142135623730951, 1.4142135623730951, 1.0, 1.4142135623730951, 1.0, 0.0,
            1.0, 0.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.4142135623730951, 1.0,
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.4142135623730951,
            1.4142135623730951, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            1.4142135623730951, 1.0, 1.0, 1.0, 1.0, 1.4142135623730951, 1.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.4142135623730951, 1.0,
            1.4142135623730951, 1.0, 0.0, 0.0, 1.0, 2.0, 2.23606797749979, 2.0,
        ];
        for (got, want) in d.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn distance_transform_edt_all_foreground_matches_real_scipy_artifact() {
        // An image with NO background pixel at all: real scipy 1.9.3
        // (the pinned CellProfiler 4.2.8 environment's
        // scipy.ndimage.distance_transform_edt) returns
        // d[r][c] = sqrt((r + 1)^2 + c^2), NOT a constant. This is an
        // algorithmic artifact of scipy's feature transform
        // (scipy/ndimage/src/ni_morphology.c): the feature array is
        // zero-initialized (scipy/ndimage/_morphology.py,
        // `ft = numpy.zeros(...)`), the base pass writes only
        // `features[0] = -1` for foreground pixels, and a `_VoronoiFT`
        // line with no background site leaves every feature untouched,
        // so every pixel measures against the stale (-1, 0) feature.
        // All values below verified against the real function.
        // 5x5 all-foreground:
        let d = distance_transform_edt(&[true; 25], 5, 5);
        #[rustfmt::skip]
        let expected: Vec<f64> = vec![
            1.0, 1.4142135623730951, 2.23606797749979, 3.1622776601683795, 4.123105625617661,
            2.0, 2.23606797749979, 2.8284271247461903, 3.605551275463989, 4.47213595499958,
            3.0, 3.1622776601683795, 3.605551275463989, 4.242640687119285, 5.0,
            4.0, 4.123105625617661, 4.47213595499958, 5.0, 5.656854249492381,
            5.0, 5.0990195135927845, 5.385164807134504, 5.830951894845301, 6.4031242374328485,
        ];
        for (got, want) in d.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        // A single row (1x5) and a single column (5x1) pin the
        // asymmetric (r + 1) / c formula in both directions:
        let d = distance_transform_edt(&[true; 5], 5, 1);
        #[rustfmt::skip]
        let expected: Vec<f64> = vec![
            1.0, 1.4142135623730951, 2.23606797749979, 3.1622776601683795, 4.123105625617661,
        ];
        for (got, want) in d.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        let d = distance_transform_edt(&[true; 5], 1, 5);
        let expected: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        for (got, want) in d.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        // The 450x450 all-foreground stress field itself (corners),
        // from the real scipy run of the empty-field pilot input:
        let d = distance_transform_edt(&vec![true; 450 * 450], 450, 450);
        assert_eq!(d[0], 1.0);
        assert_eq!(d[449], 449.00111358436516);
        assert_eq!(d[449 * 450], 450.0);
        assert_eq!(d[449 * 450 + 449], 635.6893895606564);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn distance_transform_edt_partial_columns_match_real_scipy() {
        // Columns without background but an image WITH background keep
        // their exact real-scipy values (the no-background artifact
        // only appears when the whole image is foreground). 3x3 with
        // the only background pixel at (2, 2), verified against the
        // real scipy.ndimage.distance_transform_edt:
        #[rustfmt::skip]
        let foreground: Vec<bool> = vec![
            true, true, true,
            true, true, true,
            true, true, false,
        ];
        let d = distance_transform_edt(&foreground, 3, 3);
        #[rustfmt::skip]
        let expected: Vec<f64> = vec![
            2.8284271247461903, 2.23606797749979, 2.0,
            2.23606797749979, 1.4142135623730951, 1.0,
            2.0, 1.0, 0.0,
        ];
        for (got, want) in d.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn mt19937_uniform_matches_real_numpy_legacy_random_seed_0() {
        // `numpy.random.seed(0); numpy.random.uniform(0, 0.001, (3,4))`
        // -- the exact jitter CellProfiler's Shape unclump applies.
        // Verified against real numpy.
        let mut rng = Mt19937::seed(0);
        let got = rng.uniform(0.0, 0.001, 12);
        #[rustfmt::skip]
        let expected: Vec<f64> = vec![
            0.0005488135039273248, 0.0007151893663724195, 0.0006027633760716439, 0.0005448831829968968,
            0.0004236547993389047, 0.0006458941130666561, 0.00043758721126269253, 0.0008917730007820798,
            0.0009636627605010293, 0.0003834415188257777, 0.0007917250380826646, 0.0005288949197529045,
        ];
        for (got, want) in got.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-18, "{got} vs {want}");
        }
    }

    #[test]
    fn binary_shrink_matches_real_centrosome_on_symmetric_blocks() {
        // Verified against real centrosome.cpmorphology.binary_shrink:
        // both an odd-sized block centered in its field (3x3 in 7x7)
        // and an odd-sized block with one extra all-background border
        // ring on each side (7x7 spanning rows/cols 1..=7 of a 9x9
        // field) shrink to their exact geometric center -- the 4-pass
        // directional thinning (ulr/urb/lrl/llt) is symmetric for
        // these two specific cases.
        let mut blob = vec![false; 49];
        for r in 2..5 {
            for c in 2..5 {
                blob[r * 7 + c] = true;
            }
        }
        assert_eq!(coords(&shrink(blob, 7, 7), 7), vec![(3, 3)]);

        let mut blob2 = vec![false; 81];
        for r in 1..8 {
            for c in 1..8 {
                blob2[r * 9 + c] = true;
            }
        }
        assert_eq!(coords(&shrink(blob2, 9, 9), 9), vec![(4, 4)]);
    }

    #[test]
    fn smooth_matches_real_identifyprimaryobjects_smooth_image() {
        // Fixed numpy.random.default_rng(7) 8x9 image, mask with a
        // 2x3 excluded rectangle at the top-left, filter_size =
        // 2.35*10/3.5 (automatic smoothing, minimum_diameter=10).
        // Verified against real IdentifyPrimaryObjects.smooth_image
        // (Gaussian limited to a +/-2SD kernel, mask-bleed-corrected
        // by dividing by the mask's own blurred edge weight;
        // masked-out pixels are exactly 0 in the output, matching
        // `masked_image[~mask] = 0` never being overwritten there).
        #[rustfmt::skip]
        let image: Vec<f64> = vec![
            0.625095466604667, 0.8972138009695755, 0.7756856902451935, 0.22520718999059186, 0.30016628491122543, 0.8735534453962619, 0.005265304565574724, 0.8212284183827663, 0.7970694287520462,
            0.4679349528437208, 0.3030324268193135, 0.2784256121007733, 0.2548695876541246, 0.4450763058826466, 0.5045482589579533, 0.5534973520744925, 0.9955002834343927, 0.7926619192137531,
            0.6221792294411627, 0.9889601476818849, 0.21530869823559895, 0.16021203385784455, 0.6125396042730308, 0.04394200796138337, 0.03568027877359614, 0.5148888202713703, 0.4662060253252891,
            0.9171677731928523, 0.6292262544910104, 0.5141176465995139, 0.49687343539350426, 0.24751492202733083, 0.01179402554250586, 0.19240214398531064, 0.6920321208818392, 0.2006067239869952,
            0.3695363106022067, 0.0037342420520759534, 0.8300477298017456, 0.15446108106143985, 0.26759930456378545, 0.8803321539808286, 0.5097908098684232, 0.8471502463658693, 0.6397171669425262,
            0.7417709473618571, 0.09149560506304566, 0.5411438213764888, 0.50777223630035, 0.8713393766928806, 0.3612640590141576, 0.5981840672072131, 0.05925164234550362, 0.3876318011107287,
            0.32303634625820665, 0.15019972907045187, 0.8163381038190757, 0.37944617155031246, 0.9787478844112216, 0.5899916930106103, 0.6050562538298513, 0.6379965807883322, 0.6764502438127883,
            0.1507880191683687, 0.44031346718818754, 0.23956396182952333, 0.40249829810398163, 0.09670409393174562, 0.9678280510488214, 0.21500403735588003, 0.6717651626112849, 0.3004200814790703,
        ];
        let mut mask = vec![true; 72];
        for r in 0..2 {
            for c in 0..3 {
                mask[r * 9 + c] = false;
            }
        }
        #[rustfmt::skip]
        let expected: Vec<f64> = vec![
            0.0, 0.0, 0.0, 0.3939722524515139, 0.40954411198903634, 0.4262908685163985, 0.4535670175071117, 0.4965961694824304, 0.5282746862133805,
            0.0, 0.0, 0.0, 0.40087757856699474, 0.41818187890162245, 0.4379245205467812, 0.4592676512446481, 0.5028619456572647, 0.5353966134942413,
            0.4949501980746066, 0.4704268135446195, 0.4499274006937626, 0.42032181353866793, 0.4249052769653301, 0.4409023580817412, 0.4530096859838017, 0.4859682817733337, 0.5047162239203901,
            0.4811922437320763, 0.47566199856498853, 0.4627911239467864, 0.4437629047057849, 0.4484591314739661, 0.465935967999423, 0.47031985991481523, 0.4985000746711872, 0.5063417359650657,
            0.45767022523446316, 0.46055998371453283, 0.45826078729943504, 0.45003472925705884, 0.4579124262781695, 0.4730458275012607, 0.47323565392945816, 0.4934250936069474, 0.4945058524578029,
            0.4446130960885275, 0.45377874452639927, 0.4606582357107994, 0.45720435004150783, 0.464805463660323, 0.4788412258891959, 0.47333799199659726, 0.4864712999370803, 0.48001766727925266,
            0.41718884260016226, 0.43408592924827394, 0.45841501037314986, 0.46770934769548034, 0.4868638967055498, 0.5106551784970451, 0.5038223599054911, 0.5148731271448701, 0.5087897053962643,
            0.3744748844634584, 0.4051617004194337, 0.45000528988883026, 0.47192134311628076, 0.5046911337812299, 0.5373203517807172, 0.5339026555466009, 0.5469501130148586, 0.5374002241462942,
        ];
        let result = smooth(&image, &mask, 9, 8, 2.35 * 10.0 / 3.5);
        for (got, want) in result.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
    }
}
// A deliberately non-stable min heap: equal-priority seed entries must use
// scikit-image's exact swap/pop behavior (not a Rust BinaryHeap tiebreaker).
pub(crate) struct MinHeap<T> {
    data: Vec<T>,
    smaller: fn(&T, &T) -> bool,
}
impl<T> MinHeap<T> {
    pub(crate) fn new(smaller: fn(&T, &T) -> bool) -> Self {
        Self {
            data: Vec::new(),
            smaller,
        }
    }
    pub(crate) fn push(&mut self, value: T) {
        let mut child = self.data.len();
        self.data.push(value);
        while child > 0 {
            let parent = (child - 1) / 2;
            if !(self.smaller)(&self.data[child], &self.data[parent]) {
                break;
            }
            self.data.swap(parent, child);
            child = parent;
        }
    }
    pub(crate) fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let result = self.data.swap_remove(0);
        let mut i = 0;
        loop {
            let l = 2 * i + 1;
            let r = l + 1;
            let mut smallest = i;
            if l >= self.data.len() {
                break;
            }
            if (self.smaller)(&self.data[l], &self.data[i]) {
                smallest = l;
            }
            if r < self.data.len() && (self.smaller)(&self.data[r], &self.data[smallest]) {
                smallest = r;
            }
            if smallest == i {
                break;
            }
            self.data.swap(i, smallest);
            i = smallest;
        }
        Some(result)
    }
}
#[derive(Clone, Copy)]
struct Flood {
    value: f64,
    age: usize,
    pixel: usize,
}
pub(crate) fn watershed(
    image: &[f64],
    labels: &[u32],
    markers: &[u32],
    w: usize,
    h: usize,
) -> Vec<u32> {
    let mut out = markers
        .iter()
        .zip(labels)
        .map(|(&m, &l)| if l != 0 { m } else { 0 })
        .collect::<Vec<_>>();
    let mut heap = MinHeap::new(|a: &Flood, b: &Flood| {
        if a.value != b.value {
            a.value < b.value
        } else {
            a.age < b.age
        }
    });
    for p in 0..out.len() {
        if out[p] != 0 {
            heap.push(Flood {
                value: image[p],
                age: 0,
                pixel: p,
            });
        }
    }
    let mut age = 1;
    while let Some(e) = heap.pop() {
        // skimage.morphology._offsets_to_raveled_neighbors distance order.
        for (dy, dx) in [
            (-1, 0),
            (0, -1),
            (0, 1),
            (1, 0),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ] {
            if let Some(q) = neighbor(e.pixel, w, h, dy, dx) {
                if labels[q] == 0 || out[q] != 0 {
                    continue;
                }
                out[q] = out[e.pixel];
                age += 1;
                heap.push(Flood {
                    value: image[q],
                    age,
                    pixel: q,
                });
            }
        }
    }
    out
}
#[derive(Clone, Copy)]
struct Propagation {
    key: (u64, u32, usize),
    pixel: usize,
    label: u32,
}
fn propagate(labels: &[u32], markers: &[u32], w: usize, h: usize) -> Vec<u32> {
    let mut out = vec![0; labels.len()];
    let mut distance = vec![-1.0; labels.len()];
    let mut heap = MinHeap::new(|a: &Propagation, b: &Propagation| a.key < b.key);
    for p in 0..labels.len() {
        if markers[p] > 0 {
            distance[p] = 0.0;
            if labels[p] > 0 {
                heap.push(Propagation {
                    key: (0, markers[p], p),
                    pixel: p,
                    label: markers[p],
                });
            }
        }
    }
    while let Some(e) = heap.pop() {
        if out[e.pixel] != 0 {
            continue;
        }
        out[e.pixel] = e.label;
        let d0 = distance[e.pixel];
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dy == 0 && dx == 0 {
                    continue;
                }
                if let Some(q) = neighbor(e.pixel, w, h, dy, dx) {
                    if out[q] != 0 || labels[q] == 0 {
                        continue;
                    }
                    let d = d0
                        + if dy != 0 && dx != 0 {
                            2.0_f64.sqrt()
                        } else {
                            1.0
                        };
                    if distance[q] < 0.0 || distance[q] > d {
                        distance[q] = d;
                        // centrosome converts positive doubles to two signed
                        // ints, discarding the low bit of the mantissa.
                        heap.push(Propagation {
                            key: (d.to_bits() >> 1, e.label, q),
                            pixel: q,
                            label: e.label,
                        });
                    }
                }
            }
        }
    }
    // centrosome restores even markers that fell outside the mask after
    // spline upsampling (do not silently "fix" these).
    for p in 0..markers.len() {
        if markers[p] != 0 {
            out[p] = markers[p];
        }
    }
    out
}

#[cfg(test)]
mod watershed_tests {
    use super::*;

    #[test]
    fn watershed_matches_real_skimage_segmentation_watershed() {
        // 5x5 symmetric ring with two seeds at (1,1)=1 and (3,3)=2,
        // intensity watershed_image = 1 - image. Verified against
        // real skimage.segmentation.watershed(connectivity=8,
        // markers=-labeled_maxima, mask=labeled_image!=0) negated
        // back to positive labels: seed 1 floods everything except a
        // single surviving pixel for seed 2 at (3,3) -- a degenerate
        // symmetric tie where the FIFO push/pop order of the initial
        // zero-age seeds (not just per-step tie-breaking) determines
        // the result.
        #[rustfmt::skip]
        let image: Vec<f64> = vec![
            0.9,0.9,0.9,0.9,0.9,
            0.9,0.1,0.5,0.1,0.9,
            0.9,0.1,0.5,0.1,0.9,
            0.9,0.1,0.5,0.1,0.9,
            0.9,0.9,0.9,0.9,0.9,
        ];
        let watershed_image: Vec<f64> = image.iter().map(|&v| 1.0 - v).collect();
        let labels = vec![1u32; 25];
        let mut markers = vec![0u32; 25];
        markers[5 + 1] = 1;
        markers[3 * 5 + 3] = 2;
        let out = watershed(&watershed_image, &labels, &markers, 5, 5);
        #[rustfmt::skip]
        let expected: Vec<u32> = vec![
            1,1,1,1,1,
            1,1,1,1,1,
            1,1,1,1,1,
            1,1,1,2,1,
            1,1,1,1,1,
        ];
        assert_eq!(out, expected);
    }

    #[test]
    fn propagate_matches_real_centrosome_propagate() {
        // 6x6 region mask (all True except (0,0) excluded), two
        // markers at (1,1)=1 and (4,4)=2, image=zeros (CellProfiler's
        // own Propagate watershed_method always passes a zeros image
        // -- "No image used" -- so only the geometric-distance term
        // is exercised, matching cp-anywhere's own API which omits
        // the image argument entirely). Verified against real
        // centrosome.propagate.propagate(image=zeros, markers,
        // mask=region, weight=1.0).
        let mut labels = vec![1u32; 36];
        labels[0] = 0;
        let mut markers = vec![0u32; 36];
        markers[6 + 1] = 1;
        markers[4 * 6 + 4] = 2;
        let out = propagate(&labels, &markers, 6, 6);
        #[rustfmt::skip]
        let expected: Vec<u32> = vec![
            0,1,1,1,1,1,
            1,1,1,1,1,2,
            1,1,1,1,2,2,
            1,1,1,2,2,2,
            1,1,2,2,2,2,
            1,2,2,2,2,2,
        ];
        assert_eq!(out, expected);
    }

    fn twin_blob_image() -> Vec<f64> {
        #[rustfmt::skip]
        let v = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.041685152500090084, 0.09649209709474871, 0.13103392424311155, 0.1428571428571429, 0.13103392424311155, 0.09649209709474871, 0.041685152500090084, 0.041685152500090084, 0.09649209709474871, 0.13103392424311155, 0.1428571428571429, 0.13103392424311155, 0.09649209709474871, 0.041685152500090084, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0852679660810216, 0.16700687216495702, 0.23069074183792804, 0.2715686409153165, 0.2857142857142857, 0.2715686409153165, 0.23069074183792804, 0.16700687216495702, 0.16700687216495702, 0.23069074183792804, 0.2715686409153165, 0.2857142857142857, 0.2715686409153165, 0.23069074183792804, 0.16700687216495702, 0.0852679660810216, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0852679660810216, 0.19187796435823135, 0.2857142857142857, 0.36112343500006006, 0.41098491062604847, 0.4285714285714286, 0.41098491062604847, 0.36112343500006006, 0.2857142857142857, 0.2857142857142857, 0.36112343500006006, 0.41098491062604847, 0.4285714285714286, 0.41098491062604847, 0.36112343500006006, 0.2857142857142857, 0.19187796435823135, 0.0852679660810216, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.041685152500090084, 0.16700687216495702, 0.2857142857142857, 0.3939084732686736, 0.4849212463622873, 0.5482460485473744, 0.5714285714285714, 0.5482460485473744, 0.4849212463622873, 0.3939084732686736, 0.3939084732686736, 0.4849212463622873, 0.5482460485473744, 0.5714285714285714, 0.5482460485473744, 0.4849212463622873, 0.3939084732686736, 0.2857142857142857, 0.16700687216495702, 0.041685152500090084, 0.0, 0.0,
            0.0, 0.0, 0.09649209709474871, 0.23069074183792804, 0.36112343500006006, 0.4849212463622873, 0.5959389821791157, 0.68056171750003, 0.7142857142857143, 0.68056171750003, 0.5959389821791157, 0.4849212463622873, 0.4849212463622873, 0.5959389821791157, 0.68056171750003, 0.7142857142857143, 0.68056171750003, 0.5959389821791157, 0.4849212463622873, 0.36112343500006006, 0.23069074183792804, 0.09649209709474871, 0.0, 0.0,
            0.0, 0.0, 0.13103392424311155, 0.2715686409153165, 0.41098491062604847, 0.5482460485473744, 0.68056171750003, 0.7979694910895578, 0.8571428571428572, 0.7979694910895578, 0.68056171750003, 0.5482460485473744, 0.5482460485473744, 0.68056171750003, 0.7979694910895578, 0.8571428571428572, 0.7979694910895578, 0.68056171750003, 0.5482460485473744, 0.41098491062604847, 0.2715686409153165, 0.13103392424311155, 0.0, 0.0,
            0.0, 0.0, 0.1428571428571429, 0.2857142857142857, 0.4285714285714286, 0.5714285714285714, 0.7142857142857143, 0.8571428571428572, 1.0, 0.8571428571428572, 0.7142857142857143, 0.5714285714285714, 0.5714285714285714, 0.7142857142857143, 0.8571428571428572, 1.0, 0.8571428571428572, 0.7142857142857143, 0.5714285714285714, 0.4285714285714286, 0.2857142857142857, 0.1428571428571429, 0.0, 0.0,
            0.0, 0.0, 0.13103392424311155, 0.2715686409153165, 0.41098491062604847, 0.5482460485473744, 0.68056171750003, 0.7979694910895578, 0.8571428571428572, 0.7979694910895578, 0.68056171750003, 0.5482460485473744, 0.5482460485473744, 0.68056171750003, 0.7979694910895578, 0.8571428571428572, 0.7979694910895578, 0.68056171750003, 0.5482460485473744, 0.41098491062604847, 0.2715686409153165, 0.13103392424311155, 0.0, 0.0,
            0.0, 0.0, 0.09649209709474871, 0.23069074183792804, 0.36112343500006006, 0.4849212463622873, 0.5959389821791157, 0.68056171750003, 0.7142857142857143, 0.68056171750003, 0.5959389821791157, 0.4849212463622873, 0.4849212463622873, 0.5959389821791157, 0.68056171750003, 0.7142857142857143, 0.68056171750003, 0.5959389821791157, 0.4849212463622873, 0.36112343500006006, 0.23069074183792804, 0.09649209709474871, 0.0, 0.0,
            0.0, 0.0, 0.041685152500090084, 0.16700687216495702, 0.2857142857142857, 0.3939084732686736, 0.4849212463622873, 0.5482460485473744, 0.5714285714285714, 0.5482460485473744, 0.4849212463622873, 0.3939084732686736, 0.3939084732686736, 0.4849212463622873, 0.5482460485473744, 0.5714285714285714, 0.5482460485473744, 0.4849212463622873, 0.3939084732686736, 0.2857142857142857, 0.16700687216495702, 0.041685152500090084, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0852679660810216, 0.19187796435823135, 0.2857142857142857, 0.36112343500006006, 0.41098491062604847, 0.4285714285714286, 0.41098491062604847, 0.36112343500006006, 0.2857142857142857, 0.2857142857142857, 0.36112343500006006, 0.41098491062604847, 0.4285714285714286, 0.41098491062604847, 0.36112343500006006, 0.2857142857142857, 0.19187796435823135, 0.0852679660810216, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0852679660810216, 0.16700687216495702, 0.23069074183792804, 0.2715686409153165, 0.2857142857142857, 0.2715686409153165, 0.23069074183792804, 0.16700687216495702, 0.16700687216495702, 0.23069074183792804, 0.2715686409153165, 0.2857142857142857, 0.2715686409153165, 0.23069074183792804, 0.16700687216495702, 0.0852679660810216, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.041685152500090084, 0.09649209709474871, 0.13103392424311155, 0.1428571428571429, 0.13103392424311155, 0.09649209709474871, 0.041685152500090084, 0.041685152500090084, 0.09649209709474871, 0.13103392424311155, 0.1428571428571429, 0.13103392424311155, 0.09649209709474871, 0.041685152500090084, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        v
    }

    #[test]
    fn declump_end_to_end_matches_real_identifyprimaryobjects_on_twin_blobs() {
        // Two touching circular blobs (24x20), minimum_diameter=10,
        // automatic smoothing/suppression. Verified against a direct,
        // real-source-derived reimplementation of
        // IdentifyPrimaryObjects.separate_neighboring_objects (smooth_image
        // + is_local_maximum/binary_shrink + distance_transform_edt +
        // the fixed-seed(0) jitter + skimage.segmentation.watershed),
        // for all three combinations actually exercised here:
        // Intensity/Intensity, Shape/Shape, Intensity/Shape.
        let image = twin_blob_image();
        let labels: Vec<u32> = image.iter().map(|&v| u32::from(v > 0.0)).collect();
        let base = DeclumpConfig {
            minimum_diameter: 10.0,
            low_resolution_maxima: true,
            basic: false,
            fill_holes_for_shape: false,
            smoothing: Smoothing::Automatic,
            suppression: MaximaSuppression::Automatic,
            unclump: UnclumpMethod::Intensity,
            watershed: WatershedMethod::Intensity,
        };

        let intensity_intensity = declump(&image, None, &labels, 24, 20, 1, &base).unwrap();
        #[rustfmt::skip]
        let expected_ii: Vec<u32> = vec![
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,0,0,0,0,0,
            0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,0,0,0,0,
            0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,0,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,0,0,0,
            0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        ];
        assert_eq!(
            intensity_intensity.labels, expected_ii,
            "intensity/intensity"
        );

        let intensity_shape = declump(
            &image,
            None,
            &labels,
            24,
            20,
            1,
            &DeclumpConfig {
                watershed: WatershedMethod::Shape,
                ..base.clone()
            },
        )
        .unwrap();
        #[rustfmt::skip]
        let expected_is: Vec<u32> = vec![
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,0,0,0,0,0,
            0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,0,0,0,0,
            0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,0,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,0,0,
            0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,0,0,0,
            0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        ];
        assert_eq!(intensity_shape.labels, expected_is, "intensity/shape");

        let shape_shape = declump(
            &image,
            None,
            &labels,
            24,
            20,
            1,
            &DeclumpConfig {
                unclump: UnclumpMethod::Shape,
                watershed: WatershedMethod::Shape,
                ..base
            },
        )
        .unwrap();
        #[rustfmt::skip]
        let expected_ss: Vec<u32> = vec![
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,
            0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,
            0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
            0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
            0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
            0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,
            0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        ];
        assert_eq!(shape_shape.labels, expected_ss, "shape/shape");
    }
}
