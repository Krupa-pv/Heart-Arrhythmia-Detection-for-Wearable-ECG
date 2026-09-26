import SwiftUI
import CoreML
import ProtoHead

struct Record: Codable { let rec: Int; let t: [Float]; let y: [Int]; let rr: [[Float]]; let x: [[Float]] }
struct Output: Codable {
    let rec: Int; let enroll_idx: [Int]; let test_idx: [Int]; let pred: [Int]
    let embedding: [[Float]]; let prototypes_after: [[Float]]
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
}
