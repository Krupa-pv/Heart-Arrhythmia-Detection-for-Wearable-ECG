// v3 (CNN + timing model) on one replayed record, mac check. No enrollment, every beat classified.
//
// usage: v3 <record.json> <embedding.mlpackage> <v3_constants.json> <out.json>
import CoreML
import Foundation
import ProtoHead

struct Record: Codable {
    let rec: Int
    let t: [Float]
    let y: [Int]
    let rr: [[Float]]
    let rr_norm: [[Float]]
    let x: [[Float]]
}

struct Output: Codable {
    let rec: Int
    let pred: [Int]
    let embedding: [[Float]]
}

let args = CommandLine.arguments
guard args.count >= 5 else {
    print("usage: v3 <record.json> <embedding.mlpackage> <v3_constants.json> <out.json>")
    exit(1)
}
let r = try JSONDecoder().decode(Record.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
let model = try EmbeddingModel(url: URL(fileURLWithPath: args[2]), computeUnits: .cpuOnly)
let clf = V3Classifier(constants: try V3Constants.load(URL(fileURLWithPath: args[3])))

let emb = try r.x.map { try model.embed($0) }
let pred = r.x.indices.map { clf.classify(embedding: emb[$0], rr: r.rr[$0], rrNorm: r.rr_norm[$0]) }

var correct = [Int](repeating: 0, count: 5), total = correct
for (y, p) in zip(r.y, pred) { total[y] += 1; if p == y { correct[y] += 1 } }
print("record \(r.rec): \(r.x.count) beats")
for c in 0..<5 where total[c] > 0 { print("  \(clf.c.classes[c]): \(correct[c])/\(total[c]) correct") }

try JSONEncoder().encode(Output(rec: r.rec, pred: pred, embedding: emb)).write(to: URL(fileURLWithPath: args[4]))
print("wrote \(args[4])")
