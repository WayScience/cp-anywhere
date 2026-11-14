//! Standalone CellProfiler 4.2.8 IdentifyPrimaryObjects declumping.
//! No Python, filesystem, or external-crate dependency.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnclumpMethod { None, Intensity, Shape }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WatershedMethod { None, Intensity, Shape, Propagate }
#[derive(Clone, Copy, Debug)]
pub enum Smoothing { Automatic, Manual(f64) }
#[derive(Clone, Copy, Debug)]
pub enum MaximaSuppression { Automatic, Manual(f64) }
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
        Self { unclump: UnclumpMethod::Intensity, watershed: WatershedMethod::Intensity,
            smoothing: Smoothing::Automatic, suppression: MaximaSuppression::Automatic,
            minimum_diameter: 10.0, low_resolution_maxima: true, basic: false,
            fill_holes_for_shape: false }
    }
}
#[derive(Clone, Debug)]
pub struct DeclumpResult {
    pub labels: Vec<u32>,
    pub object_count: u32,
    pub labeled_maxima: Vec<u32>,
    pub maxima_suppression_size: f64,
}
/// Arrays are row-major, `width * height` pixels. Labels are the existing
/// foreground connected components; zero is background. `object_count` is
/// preserved on the no-op path, just as CellProfiler does.
pub fn declump(image: &[f64], mask: Option<&[bool]>, labels: &[u32], width: usize,
    height: usize, object_count: u32, config: &DeclumpConfig) -> Result<DeclumpResult, String> {
    let n = width.checked_mul(height).ok_or("image dimensions overflow")?;
    if n != image.len() || n != labels.len() || mask.is_some_and(|m| m.len() != n) {
        return Err("image, labels and mask must match width * height".into());
    }
    if !config.basic && (config.unclump == UnclumpMethod::None || config.watershed == WatershedMethod::None) {
        return Ok(DeclumpResult { labels: labels.to_vec(), object_count,
            labeled_maxima: vec![0; n], maxima_suppression_size: 7.0 });
    }
    let owned_mask;
    let mask=if let Some(m)=mask {m}else{owned_mask=vec![true;n];&owned_mask};
    let smoothing=match config.smoothing {Smoothing::Automatic=>2.35*config.minimum_diameter/3.5,Smoothing::Manual(x)=>x};
    let blurred=smooth(image,mask,width,height,smoothing);
    let resize=if config.minimum_diameter>10.0 && (config.basic || config.low_resolution_maxima) {10.0/config.minimum_diameter}else{1.0};
    let suppression=if resize<1.0 {
        if config.basic || matches!(config.suppression,MaximaSuppression::Automatic) {7.0}else{match config.suppression {MaximaSuppression::Manual(x)=>x*resize+0.5,_=>unreachable!()}}
    } else {if config.basic {config.minimum_diameter/1.5}else{match config.suppression {MaximaSuppression::Automatic=>config.minimum_diameter/1.5,MaximaSuppression::Manual(x)=>x}}};
    if resize<1.0 {return Err("low resolution not yet implemented".into());}
    if !config.basic && config.unclump==UnclumpMethod::Shape {return Err("shape not yet implemented".into());}
    let binary=maxima(&blurred,labels,width,height,(suppression-0.5).max(1.0));
    let (labeled_maxima,count)=label8(&binary,width,height);
    let output=if !config.basic && config.watershed==WatershedMethod::Propagate {
        propagate(labels,&labeled_maxima,width,height)
    } else {
        if !config.basic && config.watershed==WatershedMethod::Shape {return Err("shape watershed not yet implemented".into());}
        let elevation=image.iter().map(|&x|1.0-x).collect::<Vec<_>>();
        watershed(&elevation,labels,&labeled_maxima,width,height)
    };
    Ok(DeclumpResult{labels:output,object_count:count,labeled_maxima,maxima_suppression_size:suppression/resize})
}
fn smooth(image: &[f64], mask: &[bool], w: usize, h: usize, size: f64) -> Vec<f64> {
    if size == 0.0 { return image.to_vec(); }
    let sigma = size / 2.35;
    let radius = ((size / 2.0) as usize).max(1);
    let mut weights = Vec::with_capacity(radius + 1);
    for r in 0..=radius {
        weights.push(1.0 / (2.0 * std::f64::consts::PI).sqrt() / sigma
            * (-0.5 * (r as f64).powi(2) / sigma.powi(2)).exp());
    }
    let convolve = |src: &[f64], axis: usize| {
        let mut dst = vec![0.0; src.len()];
        for y in 0..h { for x in 0..w {
            let p=y*w+x;
            let mut v=src[p]*weights[0];
            for r in (1..=radius).rev() {
                let (a,b)=if axis==0 {
                    (if y>=r {src[(y-r)*w+x]}else{0.0},if y+r<h {src[(y+r)*w+x]}else{0.0})
                } else {
                    (if x>=r {src[p-r]}else{0.0},if x+r<w {src[p+r]}else{0.0})
                };
                v += (a+b)*weights[r];
            }
            dst[p]=v;
        }}
        dst
    };
    let m=mask.iter().map(|&b|if b {1.0}else{0.0}).collect::<Vec<_>>();
    let masked=image.iter().zip(mask).map(|(&x,&b)|if b {x}else{0.0}).collect::<Vec<_>>();
    let edge=convolve(&convolve(&m,0),1);
    let blurred=convolve(&convolve(&masked,0),1);
    blurred.iter().zip(edge).zip(mask).map(|((&x,e),&b)|if b {x/e}else{0.0}).collect()
}

// Packed, bit-indexed binary_shrink lookup tables from centrosome 1.2.3
// cpmorphology.binary_shrink, in its UL/R, UR/B, LR/L, LL/T order.
const SHRINK_TABLES: [[u64;8];4] = [
    [0xffffffff7475ffff,0xffffffff74ffffff,0xffffffffffffffff,0xffffffff74ffffff,0xffffffffffffffff,0xffffffffffffffff,0xffffffffffffffff,0xffffffff74ffffff],
    [0x7f2fffff7723ffff,0x7fffffff77ffffff,0xffffffffffffffff,0xffffffffffffffff,0x7f2fffffffffffff,0x7fffffffffffffff,0xffffffffffffffff,0xffffffffffffffff],
    [0xff2effffffffffff,0xffffffffffffffff,0xffffffffffffffff,0xffffffffffffffff,0xff2efffffffeffff,0xffffffffffffffff,0xff2efffffffeffff,0xff2efffffffeffff],
    [0xffffffffffffffff,0xfffffffffcfeffff,0xfffffffffffeffff,0xfffffffffcfeffff,0xffffffffffffffff,0xffffffffffffffff,0xffeefffffffeffff,0xcceefffffcfeffff],
];
fn shrink(mut pixels: Vec<bool>, w: usize, h: usize) -> Vec<bool> {
    let mut active=pixels.iter().enumerate().filter_map(|(i,&b)|b.then_some(i)).collect::<Vec<_>>();
    loop {
        let count=active.len();
        for table in SHRINK_TABLES {
            let mut remove=Vec::new();
            for &p in &active {
                let mut index=0usize;
                for dy in -1..=1 { for dx in -1..=1 {
                    if let Some(q)=neighbor(p,w,h,dy,dx) {if pixels[q] {index |= 1 << ((dy+1)*3+dx+1);}}
                }}
                if table[index/64] & (1 << (index%64))==0 {remove.push(p);}
            }
            for p in remove { pixels[p]=false; }
            active.retain(|&p|pixels[p]);
        }
        if active.len()==count {break;}
    }
    pixels
}
fn neighbor(p: usize,w: usize,h: usize,dy: isize,dx: isize) -> Option<usize> {
    let y=(p/w).checked_add_signed(dy)?;
    let x=(p%w).checked_add_signed(dx)?;
    (y<h && x<w).then_some(y*w+x)
}
fn label8(binary: &[bool], w: usize, h: usize) -> (Vec<u32>,u32) {
    let mut out=vec![0;binary.len()];let mut count=0;
    let mut todo=Vec::new();
    for p in 0..binary.len() {
        if !binary[p] || out[p]!=0 {continue;}
        count+=1;out[p]=count;todo.push(p);
        while let Some(q)=todo.pop() {
            for dy in -1..=1 {for dx in -1..=1 {
                if let Some(r)=neighbor(q,w,h,dy,dx) {
                    if binary[r] && out[r]==0 {out[r]=count;todo.push(r);}
                }
            }}
        }
    }
    (out,count)
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
        let labels: Vec<u32> = (0..144)
            .map(|p| if p / 12 < 2 { 0 } else { 1 })
            .collect();
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
struct MinHeap<T> { data: Vec<T>, smaller: fn(&T,&T)->bool }
impl<T> MinHeap<T> {
    fn new(smaller: fn(&T,&T)->bool) -> Self {Self {data:Vec::new(),smaller}}
    fn push(&mut self, value: T) {
        let mut child=self.data.len();self.data.push(value);
        while child>0 {let parent=(child-1)/2;
            if !(self.smaller)(&self.data[child],&self.data[parent]) {break;}
            self.data.swap(parent,child);child=parent;
        }
    }
    fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {return None;}
        let result=self.data.swap_remove(0);let mut i=0;
        loop {
            let l=2*i+1;let r=l+1;let mut smallest=i;
            if l>=self.data.len() {break;}
            if (self.smaller)(&self.data[l],&self.data[i]) {smallest=l;}
            if r<self.data.len() && (self.smaller)(&self.data[r],&self.data[smallest]) {smallest=r;}
            if smallest==i {break;}
            self.data.swap(i,smallest);i=smallest;
        }
        Some(result)
    }
}
#[derive(Clone, Copy)]
struct Flood { value: f64, age: usize, pixel: usize }
fn watershed(image: &[f64], labels: &[u32], markers: &[u32], w: usize,h: usize) -> Vec<u32> {
    let mut out=markers.iter().zip(labels).map(|(&m,&l)|if l!=0 {m}else{0}).collect::<Vec<_>>();
    let mut heap=MinHeap::new(|a:&Flood,b:&Flood|if a.value!=b.value {a.value<b.value}else{a.age<b.age});
    for p in 0..out.len() {if out[p]!=0 {heap.push(Flood{value:image[p],age:0,pixel:p});}}
    let mut age=1;
    while let Some(e)=heap.pop() {
        // skimage.morphology._offsets_to_raveled_neighbors distance order.
        for (dy,dx) in [(-1,0),(0,-1),(0,1),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)] {
            if let Some(q)=neighbor(e.pixel,w,h,dy,dx) {
                if labels[q]==0 || out[q]!=0 {continue;}
                out[q]=out[e.pixel];age+=1;
                heap.push(Flood{value:image[q],age,pixel:q});
            }
        }
    }
    out
}
#[derive(Clone, Copy)]
struct Propagation { key: (u64,u32,usize), pixel: usize, label: u32 }
fn propagate(labels: &[u32],markers: &[u32],w: usize,h: usize) -> Vec<u32> {
    let mut out=vec![0;labels.len()];let mut distance=vec![-1.0;labels.len()];
    let mut heap=MinHeap::new(|a:&Propagation,b:&Propagation|a.key<b.key);
    for p in 0..labels.len() {
        if markers[p]>0 {distance[p]=0.0;
            if labels[p]>0 {heap.push(Propagation{key:(0,markers[p],p),pixel:p,label:markers[p]});}
        }
    }
    while let Some(e)=heap.pop() {
        if out[e.pixel]!=0 {continue;}
        out[e.pixel]=e.label;
        let d0=distance[e.pixel];
        for dy in -1..=1 {for dx in -1..=1 {
            if dy==0 && dx==0 {continue;}
            if let Some(q)=neighbor(e.pixel,w,h,dy,dx) {
                if out[q]!=0 || labels[q]==0 {continue;}
                let d=d0+if dy!=0 && dx!=0 {2.0_f64.sqrt()}else{1.0};
                if distance[q]<0.0 || distance[q]>d {
                    distance[q]=d;
                    // centrosome converts positive doubles to two signed
                    // ints, discarding the low bit of the mantissa.
                    heap.push(Propagation{key:(d.to_bits()>>1,e.label,q),pixel:q,label:e.label});
                }
            }
        }}
    }
    // centrosome restores even markers that fell outside the mask after
    // spline upsampling (do not silently "fix" these).
    for p in 0..markers.len() {if markers[p]!=0 {out[p]=markers[p];}}
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
        markers[1 * 5 + 1] = 1;
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
        markers[1 * 6 + 1] = 1;
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
}

