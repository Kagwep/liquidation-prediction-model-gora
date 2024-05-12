use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 1297, sign: true });
a.append(FP16x16 { mag: 1909, sign: false });
a.append(FP16x16 { mag: 2353, sign: false });
a.append(FP16x16 { mag: 4691, sign: false });
a.append(FP16x16 { mag: 4542, sign: false });
a.append(FP16x16 { mag: 5624, sign: true });
a.append(FP16x16 { mag: 1289, sign: true });
a.append(FP16x16 { mag: 14798, sign: false });
}