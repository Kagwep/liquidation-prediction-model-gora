use orion::numbers::{FixedTrait, FP16x16};

fn compute(ref a: Array<FP16x16>) {
a.append(FP16x16 { mag: 12045, sign: false });
a.append(FP16x16 { mag: 66246, sign: true });
a.append(FP16x16 { mag: 25002, sign: true });
a.append(FP16x16 { mag: 43941, sign: false });
a.append(FP16x16 { mag: 27885, sign: true });
a.append(FP16x16 { mag: 95845, sign: true });
a.append(FP16x16 { mag: 135390, sign: true });
a.append(FP16x16 { mag: 31049, sign: true });
a.append(FP16x16 { mag: 8260, sign: false });
a.append(FP16x16 { mag: 551, sign: false });
a.append(FP16x16 { mag: 10605, sign: true });
a.append(FP16x16 { mag: 104126, sign: true });
a.append(FP16x16 { mag: 41061, sign: true });
a.append(FP16x16 { mag: 45971, sign: true });
a.append(FP16x16 { mag: 32663, sign: false });
a.append(FP16x16 { mag: 3416, sign: false });
a.append(FP16x16 { mag: 26708, sign: true });
a.append(FP16x16 { mag: 52159, sign: true });
a.append(FP16x16 { mag: 83661, sign: true });
a.append(FP16x16 { mag: 52153, sign: true });
a.append(FP16x16 { mag: 39157, sign: true });
a.append(FP16x16 { mag: 22557, sign: false });
a.append(FP16x16 { mag: 34162, sign: true });
a.append(FP16x16 { mag: 28713, sign: true });
a.append(FP16x16 { mag: 42698, sign: true });
a.append(FP16x16 { mag: 83616, sign: true });
a.append(FP16x16 { mag: 141988, sign: true });
a.append(FP16x16 { mag: 38089, sign: true });
a.append(FP16x16 { mag: 79244, sign: true });
a.append(FP16x16 { mag: 106882, sign: true });
a.append(FP16x16 { mag: 71622, sign: true });
a.append(FP16x16 { mag: 29535, sign: true });
a.append(FP16x16 { mag: 52699, sign: true });
a.append(FP16x16 { mag: 25122, sign: true });
a.append(FP16x16 { mag: 31783, sign: true });
a.append(FP16x16 { mag: 24935, sign: false });
a.append(FP16x16 { mag: 50973, sign: true });
a.append(FP16x16 { mag: 24704, sign: true });
a.append(FP16x16 { mag: 71043, sign: true });
a.append(FP16x16 { mag: 46697, sign: true });
a.append(FP16x16 { mag: 22014, sign: true });
a.append(FP16x16 { mag: 85544, sign: true });
a.append(FP16x16 { mag: 69027, sign: true });
a.append(FP16x16 { mag: 37893, sign: true });
a.append(FP16x16 { mag: 23811, sign: false });
a.append(FP16x16 { mag: 29521, sign: true });
a.append(FP16x16 { mag: 13444, sign: true });
a.append(FP16x16 { mag: 78641, sign: true });
a.append(FP16x16 { mag: 33262, sign: true });
a.append(FP16x16 { mag: 84103, sign: true });
a.append(FP16x16 { mag: 97712, sign: true });
a.append(FP16x16 { mag: 23125, sign: true });
a.append(FP16x16 { mag: 23544, sign: true });
a.append(FP16x16 { mag: 152383, sign: true });
a.append(FP16x16 { mag: 58209, sign: true });
a.append(FP16x16 { mag: 49834, sign: true });
a.append(FP16x16 { mag: 186075, sign: true });
a.append(FP16x16 { mag: 84165, sign: true });
a.append(FP16x16 { mag: 18251, sign: true });
a.append(FP16x16 { mag: 51414, sign: true });
a.append(FP16x16 { mag: 7472, sign: false });
a.append(FP16x16 { mag: 49056, sign: true });
a.append(FP16x16 { mag: 22178, sign: false });
a.append(FP16x16 { mag: 36037, sign: true });
a.append(FP16x16 { mag: 83472, sign: true });
a.append(FP16x16 { mag: 71169, sign: true });
a.append(FP16x16 { mag: 19085, sign: true });
a.append(FP16x16 { mag: 51306, sign: true });
a.append(FP16x16 { mag: 92586, sign: true });
a.append(FP16x16 { mag: 52880, sign: true });
a.append(FP16x16 { mag: 22075, sign: false });
a.append(FP16x16 { mag: 63815, sign: true });
a.append(FP16x16 { mag: 76550, sign: true });
a.append(FP16x16 { mag: 40218, sign: true });
a.append(FP16x16 { mag: 48163, sign: true });
a.append(FP16x16 { mag: 42117, sign: true });
a.append(FP16x16 { mag: 19550, sign: false });
a.append(FP16x16 { mag: 38360, sign: true });
a.append(FP16x16 { mag: 112152, sign: true });
a.append(FP16x16 { mag: 85728, sign: true });
a.append(FP16x16 { mag: 67053, sign: true });
a.append(FP16x16 { mag: 11905, sign: false });
a.append(FP16x16 { mag: 37205, sign: true });
a.append(FP16x16 { mag: 48578, sign: true });
a.append(FP16x16 { mag: 37503, sign: false });
a.append(FP16x16 { mag: 24353, sign: true });
a.append(FP16x16 { mag: 29297, sign: true });
a.append(FP16x16 { mag: 50144, sign: true });
a.append(FP16x16 { mag: 81038, sign: true });
a.append(FP16x16 { mag: 27602, sign: true });
a.append(FP16x16 { mag: 25009, sign: false });
a.append(FP16x16 { mag: 64607, sign: true });
a.append(FP16x16 { mag: 41051, sign: false });
a.append(FP16x16 { mag: 52367, sign: true });
a.append(FP16x16 { mag: 35325, sign: true });
a.append(FP16x16 { mag: 88713, sign: true });
a.append(FP16x16 { mag: 30202, sign: true });
a.append(FP16x16 { mag: 95504, sign: true });
a.append(FP16x16 { mag: 123251, sign: true });
a.append(FP16x16 { mag: 52409, sign: true });
a.append(FP16x16 { mag: 87537, sign: true });
a.append(FP16x16 { mag: 19399, sign: false });
a.append(FP16x16 { mag: 171788, sign: true });
a.append(FP16x16 { mag: 40556, sign: true });
a.append(FP16x16 { mag: 28214, sign: true });
a.append(FP16x16 { mag: 44339, sign: true });
a.append(FP16x16 { mag: 89170, sign: true });
a.append(FP16x16 { mag: 14096, sign: false });
a.append(FP16x16 { mag: 161750, sign: true });
a.append(FP16x16 { mag: 128986, sign: true });
a.append(FP16x16 { mag: 25536, sign: true });
a.append(FP16x16 { mag: 59588, sign: false });
a.append(FP16x16 { mag: 89054, sign: true });
a.append(FP16x16 { mag: 27516, sign: true });
a.append(FP16x16 { mag: 3828, sign: false });
a.append(FP16x16 { mag: 12933, sign: true });
a.append(FP16x16 { mag: 55826, sign: true });
a.append(FP16x16 { mag: 10362, sign: false });
a.append(FP16x16 { mag: 82909, sign: true });
a.append(FP16x16 { mag: 155744, sign: true });
a.append(FP16x16 { mag: 52905, sign: true });
a.append(FP16x16 { mag: 76136, sign: true });
a.append(FP16x16 { mag: 67534, sign: true });
a.append(FP16x16 { mag: 75584, sign: true });
a.append(FP16x16 { mag: 67716, sign: true });
a.append(FP16x16 { mag: 58379, sign: true });
a.append(FP16x16 { mag: 60888, sign: true });
a.append(FP16x16 { mag: 38561, sign: true });
}