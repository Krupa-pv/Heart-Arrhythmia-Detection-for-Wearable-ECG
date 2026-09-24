import CoreML
import Foundation

/// constants from runs/enroll/proto_constants.json
public struct ProtoConstants: Codable {
    public let alpha: Float
    public let prototypes: [[Float]]  // 5 x 68, class order N S V F Q
    public let scale: [Float]         // 68, DS1 train std
    public let classes: [String]

    public static func load(_ url: URL) throws -> ProtoConstants {
        try JSONDecoder().decode(ProtoConstants.self, from: Data(contentsOf: url))
    }
}

/// Nearest prototype classifier on [64-d embedding, 4 RR] / scale.
/// Enrollment at rest = every enrollment beat is N, so only the N prototype moves,
/// blended toward the mean of the enrollment vectors by alpha. same as enroll.py
public struct ProtoHead {
    public let constants: ProtoConstants
    public private(set) var prototypes: [[Float]]

    public init(constants: ProtoConstants) {
        self.constants = constants
        self.prototypes = constants.prototypes
    }

    public static func features(embedding: [Float], rr: [Float]) -> [Float] {
        embedding + rr
    }

    /// always starts from the population prototypes, so calling it twice doesn't stack
    public mutating func enrollAtRest(_ vectors: [[Float]], nClass: Int = 0) {
        precondition(!vectors.isEmpty, "need at least one enrollment beat")
        let dim = constants.scale.count
        var mean = [Double](repeating: 0, count: dim)
        for v in vectors { for j in 0..<dim { mean[j] += Double(v[j]) } }
        let a = constants.alpha
        prototypes = constants.prototypes
        for j in 0..<dim {
            let m = Float(mean[j] / Double(vectors.count))
            prototypes[nClass][j] = (1 - a) * prototypes[nClass][j] + a * m
        }
    }

    public mutating func reset() { prototypes = constants.prototypes }

    public func classify(_ v: [Float]) -> Int {
        var best = 0, bestD = Float.infinity
        for (c, p) in prototypes.enumerated() {
            var d: Float = 0
            for j in 0..<v.count {
                let diff = (v[j] - p[j]) / constants.scale[j]
                d += diff * diff
            }
            if d < bestD { best = c; bestD = d }
        }
        return best
    }
}

/// wraps the Core ML embedding model (beat [1,256] -> embedding [1,64]), batch 1
public final class EmbeddingModel {
    let model: MLModel
    let input: MLMultiArray

    /// url can be a .mlpackage (compiled here) or an already compiled .mlmodelc
    public init(url: URL, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let compiled = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
        model = try MLModel(contentsOf: compiled, configuration: config)
        input = try MLMultiArray(shape: [1, 256], dataType: .float32)
    }

    public func embed(_ beat: [Float]) throws -> [Float] {
        precondition(beat.count == 256, "beat must be 256 samples")
        let ptr = input.dataPointer.bindMemory(to: Float.self, capacity: 256)
        for i in 0..<256 { ptr[i] = beat[i] }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["beat": input]))
        guard let e = out.featureValue(for: "embedding")?.multiArrayValue else {
            throw NSError(domain: "ProtoHead", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "no embedding output"])
        }
        // output is fp16 in the fp16 model, NSNumber indexing handles any type
        return (0..<e.count).map { e[$0].floatValue }
    }
}
