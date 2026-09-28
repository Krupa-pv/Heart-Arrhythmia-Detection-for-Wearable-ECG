import SwiftUI
import CoreML
import ProtoHead

struct Record: Codable {
    let rec: Int; let t: [Float]; let y: [Int]; let rr: [[Float]]; let x: [[Float]]
    let rr_norm: [[Float]]?  // only needed by v3
}
struct V3Output: Codable { let rec: Int; let pred: [Int]; let embedding: [[Float]] }
struct Output: Codable {
    let rec: Int; let enroll_idx: [Int]; let test_idx: [Int]; let pred: [Int]
    let embedding: [[Float]]; let prototypes_after: [[Float]]
}
// same format as the mac finetune tool, so replay.py finetune can check it
struct Cond: Codable { let pred: [Int]; let weights: [Float]; let seconds: Double }
struct FinetuneOutput: Codable {
    let rec: Int; let test_idx: [Int]; let embedding: [[Float]]; let population: [Int]; let conds: [String: Cond]
}

struct ContentView: View {
    @State private var log = "tap run"
    @State private var cpuOnly = true

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle("CPU only (should match the Mac exactly)", isOn: $cpuOnly)
            Button("Run replay, record 214") {
                log = "running..."
                Task {
                    await Task.yield()
                    do { log = try replay() } catch { log = "error: \(error)" }
                }
            }
            Button("Run MLUpdateTask fine-tune, record 214") {
                log = "running..."
                Task {
                    await Task.yield()
                    do { log = try await finetune() } catch { log = "error: \(error)" }
                }
            }
            Button("Run v3 (CNN + timing model), record 214") {
                log = "running..."
                Task {
                    await Task.yield()
                    do { log = try runV3() } catch { log = "error: \(error)" }
                }
            }
            ScrollView { Text(log).font(.system(.footnote, design: .monospaced)) }
        }
        .padding()
    }

    func file(_ name: String, _ ext: String) -> URL {
        Bundle.main.url(forResource: name, withExtension: ext)!
    }

    func ms(_ d: Duration) -> Double {
        Double(d.components.seconds) * 1000 + Double(d.components.attoseconds) / 1e15
    }

    func replay() throws -> String {
        let clock = ContinuousClock()
        let r = try JSONDecoder().decode(Record.self, from: Data(contentsOf: file("214", "json")))

        var start = clock.now
        let model = try EmbeddingModel(url: file("ecg_embedding_fp16", "mlmodelc"),
                                       computeUnits: cpuOnly ? .cpuOnly : .all)
        let loadMs = ms(clock.now - start)
        var head = ProtoHead(constants: try ProtoConstants.load(file("proto_constants", "json")))

        let t0 = r.t.min()!
        let enrollIdx = r.t.indices.filter { r.t[$0] < t0 + 60 }
        let testIdx = r.t.indices.filter { r.t[$0] >= t0 + 300 }

        // enrollment = embed the first 60 s of beats + move the N prototype
        start = clock.now
        let enrollFeats = try enrollIdx.map { i in
            ProtoHead.features(embedding: try model.embed(r.x[i]), rr: r.rr[i])
        }
        head.enrollAtRest(enrollFeats)
        let enrollMs = ms(clock.now - start)

        // embed every beat (for the check file), classify the test beats
        start = clock.now
        let emb = try r.x.map { try model.embed($0) }
        let perBeatMs = ms(clock.now - start) / Double(r.x.count)
        let pred = testIdx.map { head.classify(ProtoHead.features(embedding: emb[$0], rr: r.rr[$0])) }

        var correct = [Int](repeating: 0, count: 5), total = correct
        for (i, p) in zip(testIdx, pred) { total[r.y[i]] += 1; if p == r.y[i] { correct[r.y[i]] += 1 } }

        let out = Output(rec: r.rec, enroll_idx: enrollIdx, test_idx: testIdx, pred: pred,
                         embedding: emb, prototypes_after: head.prototypes)
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try JSONEncoder().encode(out).write(to: docs.appendingPathComponent("214_swift.json"))

        var s = "record \(r.rec): \(r.x.count) beats, enrolled on \(enrollIdx.count), tested on \(testIdx.count)\n"
        s += "compute units: \(cpuOnly ? "CPU only" : "all")\n"
        s += String(format: "model load %.1f ms\nenrollment %.1f ms (%d beats)\nembed %.3f ms per beat\n",
                    loadMs, enrollMs, enrollIdx.count, perBeatMs)
        for c in 0..<5 where total[c] > 0 {
            s += "\(head.constants.classes[c]): \(correct[c])/\(total[c]) correct\n"
        }
        return s + "saved 214_swift.json to Documents"
    }

    // v3: CNN + timing model on every beat, no enrollment
    func runV3() throws -> String {
        let r = try JSONDecoder().decode(Record.self, from: Data(contentsOf: file("214", "json")))
        guard let rrNorm = r.rr_norm else { return "214.json has no rr_norm, re-export it with replay.py" }
        let model = try EmbeddingModel(url: file("ecg_embedding_v3_fp16", "mlmodelc"), computeUnits: .cpuOnly)
        let clf = V3Classifier(constants: try V3Constants.load(file("v3_constants", "json")))

        let clock = ContinuousClock()
        let start = clock.now
        let emb = try r.x.map { try model.embed($0) }
        let pred = r.x.indices.map { clf.classify(embedding: emb[$0], rr: r.rr[$0], rrNorm: rrNorm[$0]) }
        let perBeat = ms(clock.now - start) / Double(r.x.count)

        var correct = [Int](repeating: 0, count: 5), total = correct
        for (y, p) in zip(r.y, pred) { total[y] += 1; if p == y { correct[y] += 1 } }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try JSONEncoder().encode(V3Output(rec: r.rec, pred: pred, embedding: emb))
            .write(to: docs.appendingPathComponent("214_v3.json"))

        var s = "record \(r.rec), v3, all \(r.x.count) beats\n"
        s += String(format: "%.3f ms per beat (embed + CNN head + timing model)\n", perBeat)
        for c in 0..<5 where total[c] > 0 { s += "\(clf.c.classes[c]): \(correct[c])/\(total[c]) correct\n" }
        s += "\nmac got: N 1704/2001, V 244/256, F 0/1, Q 0/2\n"
        return s + "saved 214_v3.json to Documents"
    }

    // last-layer fine-tune on the phone: realistic@60 (all N) and oracle@300, 30 SGD steps
    func finetune() async throws -> String {
        let r = try JSONDecoder().decode(Record.self, from: Data(contentsOf: file("214", "json")))
        let model = try EmbeddingModel(url: file("ecg_embedding_fp16", "mlmodelc"), computeUnits: .cpuOnly)
        let head = try UpdatableHead(url: file("ecg_head_updatable", "mlmodelc"))

        let emb = try r.x.map { try model.embed($0) }
        let feats = zip(emb, r.rr).map { ProtoHead.features(embedding: $0, rr: $1) }
        let t0 = r.t.min()!
        let testIdx = r.t.indices.filter { r.t[$0] >= t0 + 300 }
        let base = try head.baseModel()
        let population = try testIdx.map { try UpdatableHead.classify(base, feats[$0]) }

        var s = "record \(r.rec), MLUpdateTask on the head, tested on \(testIdx.count) beats\n"
        var conds: [String: Cond] = [:]
        for (name, secs, oracle) in [("realistic@60", Float(60), false), ("oracle@300", Float(300), true)] {
            let idx = r.t.indices.filter { r.t[$0] < t0 + secs }
            let clock = ContinuousClock()
            let start = clock.now
            let m = try await head.finetune(features: idx.map { feats[$0] }, labels: idx.map { oracle ? r.y[$0] : 0 })
            let took = clock.now - start
            let pred = try testIdx.map { try UpdatableHead.classify(m, feats[$0]) }
            conds[name] = Cond(pred: pred, weights: try UpdatableHead.weights(m), seconds: ms(took) / 1000)

            var correct = [Int](repeating: 0, count: 5), total = correct
            for (i, p) in zip(testIdx, pred) { total[r.y[i]] += 1; if p == r.y[i] { correct[r.y[i]] += 1 } }
            s += String(format: "\n%@: update %.1f ms on %d beats\n", name, ms(took), idx.count)
            for c in 0..<5 where total[c] > 0 { s += "  \(["N", "S", "V", "F", "Q"][c]): \(correct[c])/\(total[c]) correct\n" }
        }
        let out = FinetuneOutput(rec: r.rec, test_idx: testIdx, embedding: emb, population: population, conds: conds)
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try JSONEncoder().encode(out).write(to: docs.appendingPathComponent("214_finetune.json"))
        s += "\nmac got: realistic@60 N 1660/1662 V 128/212, oracle@300 N 1655/1662 V 138/212\n"
        return s + "saved 214_finetune.json to Documents"
    }
}
