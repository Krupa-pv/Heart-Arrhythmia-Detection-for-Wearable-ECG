// Replays one MIT-BIH record (from replay.py export) beat by beat:
// embed with Core ML -> enroll at rest on the first 60 s -> classify every beat after 300 s.
//
// usage: replay <record.json> <model.mlpackage> <proto_constants.json> <out.json> [cpu|all]
import CoreML
import Foundation
import ProtoHead

struct Record: Codable {
    let rec: Int
    let t: [Float]
    let y: [Int]
    let rr: [[Float]]
    let x: [[Float]]
}

struct Output: Codable {
    let rec: Int
    let enroll_idx: [Int]
    let test_idx: [Int]
    let pred: [Int]
    let embedding: [[Float]]
    let prototypes_after: [[Float]]
}

let args = CommandLine.arguments
guard args.count >= 5 else {
    print("usage: replay <record.json> <model.mlpackage> <proto_constants.json> <out.json> [cpu|all]")
    exit(1)
}
let enrollSec: Float = 60, testStartSec: Float = 300
let units: MLComputeUnits = (args.count > 5 && args[5] == "all") ? .all : .cpuOnly

let r = try JSONDecoder().decode(Record.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
let model = try EmbeddingModel(url: URL(fileURLWithPath: args[2]), computeUnits: units)
var head = ProtoHead(constants: try ProtoConstants.load(URL(fileURLWithPath: args[3])))

// every beat goes through the model once, like it would streaming in
let emb = try r.x.map { try model.embed($0) }
let feats = zip(emb, r.rr).map { ProtoHead.features(embedding: $0, rr: $1) }

let t0 = r.t.min()!
let enrollIdx = r.t.indices.filter { r.t[$0] < t0 + enrollSec }
let testIdx = r.t.indices.filter { r.t[$0] >= t0 + testStartSec }
precondition(enrollIdx.max()! < testIdx.min()!, "enrollment beats overlap test beats")

head.enrollAtRest(enrollIdx.map { feats[$0] })
let pred = testIdx.map { head.classify(feats[$0]) }

// quick summary, python does the real scoring
let classes = head.constants.classes
var correct = [Int](repeating: 0, count: 5), total = [Int](repeating: 0, count: 5)
for (i, p) in zip(testIdx, pred) { total[r.y[i]] += 1; if p == r.y[i] { correct[r.y[i]] += 1 } }
print("record \(r.rec): \(r.x.count) beats, enrolled on \(enrollIdx.count), tested on \(testIdx.count)")
for c in 0..<5 where total[c] > 0 {
    print("  \(classes[c]): \(correct[c])/\(total[c]) correct")
}

let out = Output(rec: r.rec, enroll_idx: enrollIdx, test_idx: testIdx, pred: pred,
                 embedding: emb, prototypes_after: head.prototypes)
try JSONEncoder().encode(out).write(to: URL(fileURLWithPath: args[4]))
print("wrote \(args[4])")
