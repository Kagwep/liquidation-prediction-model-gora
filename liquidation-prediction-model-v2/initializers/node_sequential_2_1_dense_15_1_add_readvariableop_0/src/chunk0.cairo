use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 207055, sign: true });
a.append(FP16x16 { mag: 38988, sign: true });
a.append(FP16x16 { mag: 179565, sign: true });
a.append(FP16x16 { mag: 21793, sign: true });
a.append(FP16x16 { mag: 198884, sign: false });
a.append(FP16x16 { mag: 140498, sign: false });
a.append(FP16x16 { mag: 31634, sign: true });
a.append(FP16x16 { mag: 148962, sign: true });
a.append(FP16x16 { mag: 44520, sign: true });
a.append(FP16x16 { mag: 46849, sign: true });
a.append(FP16x16 { mag: 173510, sign: false });
a.append(FP16x16 { mag: 43390, sign: true });
a.append(FP16x16 { mag: 62974, sign: true });
a.append(FP16x16 { mag: 123941, sign: false });
a.append(FP16x16 { mag: 165816, sign: true });
a.append(FP16x16 { mag: 226250, sign: true });
}