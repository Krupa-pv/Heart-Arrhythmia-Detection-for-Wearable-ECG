// Last-layer fine-tune with MLUpdateTask on one replayed record (mac check).
// embed every beat -> realistic@60 (all N) and oracle@300 (true labels) -> classify beats after 300 s
//
// usage: finetune <record.json> <embedding.mlpackage> <head.mlmodel> <out.json>
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

struct Cond: Codable {
    let pred: [Int]
    let weights: [Float]
    let seconds: Double
}

struct Output: Codable {
    let rec: Int
    let test_idx: [Int]
    let embedding: [[Float]]
    let population: [Int]
    let conds: [String: Cond]
}

let args = CommandLine.arguments
guard args.count >= 5 else {
    print("usage: finetune <record.json> <embedding.mlpackage> <head.mlmodel> <out.json>")
    exit(1)
}
let r = try JSONDecoder().decode(Record.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
let emb = try EmbeddingModel(url: URL(fileURLWithPath: args[2]), computeUnits: .cpuOnly)
let head = try UpdatableHead(url: URL(fileURLWithPath: args[3]))

let e = try r.x.map { try emb.embed($0) }
let feats = zip(e, r.rr).map { ProtoHead.features(embedding: $0, rr: $1) }
let t0 = r.t.min()!
let testIdx = r.t.indices.filter { r.t[$0] >= t0 + 300 }

let base = try head.baseModel()
let population = try testIdx.map { try UpdatableHead.classify(base, feats[$0]) }

var conds: [String: Cond] = [:]
for (name, secs, oracle) in [("realistic@60", Float(60), false), ("oracle@300", Float(300), true)] {
    let idx = r.t.indices.filter { r.t[$0] < t0 + secs }
    let labels = idx.map { oracle ? r.y[$0] : 0 }
    let start = Date()
    let m = try await head.finetune(features: idx.map { feats[$0] }, labels: labels)
    let secsTaken = Date().timeIntervalSince(start)
    let pred = try testIdx.map { try UpdatableHead.classify(m, feats[$0]) }
    conds[name] = Cond(pred: pred, weights: try UpdatableHead.weights(m), seconds: secsTaken)
    print(String(format: "%@: %d enrollment beats, update took %.1f ms", name, idx.count, secsTaken * 1000))
}

let out = Output(rec: r.rec, test_idx: testIdx, embedding: e, population: population, conds: conds)
try JSONEncoder().encode(out).write(to: URL(fileURLWithPath: args[4]))
print("wrote \(args[4])")
