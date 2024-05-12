use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 183789, sign: true });
a.append(FP16x16 { mag: 320303, sign: true });
a.append(FP16x16 { mag: 146827, sign: true });
a.append(FP16x16 { mag: 199012, sign: false });
a.append(FP16x16 { mag: 91304, sign: false });
a.append(FP16x16 { mag: 381988, sign: false });
a.append(FP16x16 { mag: 177716, sign: true });
a.append(FP16x16 { mag: 18295, sign: false });
}