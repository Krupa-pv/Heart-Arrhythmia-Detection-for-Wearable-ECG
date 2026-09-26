import CoreML
import Foundation

/// The linear head as an updatable Core ML model (ecg_head_updatable.mlmodel from updatable.py).
/// finetune() runs MLUpdateTask: plain SGD + cross entropy, full batch, same as enroll.py.
public final class UpdatableHead {
    let compiledURL: URL

    /// url can be the .mlmodel (compiled here) or an already compiled .mlmodelc
    public init(url: URL) throws {
        compiledURL = url.pathExtension == "mlmodelc" ? url : try MLModel.compileModel(at: url)
    }

    /// the head before any update
    public func baseModel() throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        return try MLModel(contentsOf: compiledURL, configuration: config)
    }

    /// full batch, so one epoch = one SGD step. returns the updated model (not saved to disk)
    public func finetune(features: [[Float]], labels: [Int], steps: Int = 30,
                         learningRate: Double = 0.01) async throws -> MLModel {
        precondition(features.count == labels.count && !features.isEmpty)
        var providers: [MLFeatureProvider] = []
        for (f, y) in zip(features, labels) {
            let x = try MLMultiArray(shape: [NSNumber(value: f.count)], dataType: .double)
            for j in 0..<f.count { x[j] = NSNumber(value: f[j]) }
            let t = try MLMultiArray(shape: [1], dataType: .int32)
            t[0] = NSNumber(value: y)
            providers.append(try MLDictionaryFeatureProvider(
                dictionary: ["features": MLFeatureValue(multiArray: x), "probs_true": MLFeatureValue(multiArray: t)]))
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        config.parameters = [.miniBatchSize: features.count, .epochs: steps, .learningRate: learningRate,
                             .shuffle: false]
        let url = compiledURL
        return try await withCheckedThrowingContinuation { cont in
            do {
                let task = try MLUpdateTask(forModelAt: url, trainingData: MLArrayBatchProvider(array: providers),
                                            configuration: config) { ctx in
                    if let e = ctx.task.error { cont.resume(throwing: e) } else { cont.resume(returning: ctx.model) }
                }
                task.resume()
            } catch { cont.resume(throwing: error) }
        }
    }

    public static func classify(_ model: MLModel, _ v: [Float]) throws -> Int {
        let x = try MLMultiArray(shape: [NSNumber(value: v.count)], dataType: .double)
        for j in 0..<v.count { x[j] = NSNumber(value: v[j]) }
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["features": x]))
        let p = out.featureValue(for: "probs")!.multiArrayValue!
        var best = 0
        for c in 1..<p.count where p[c].doubleValue > p[best].doubleValue { best = c }
        return best
    }

    /// weights of the fc layer, flattened (5 x 68), to compare with pytorch
    public static func weights(_ model: MLModel) throws -> [Float] {
        let w = try model.parameterValue(for: MLParameterKey.weights.scoped(to: "fc")) as! MLMultiArray
        return (0..<w.count).map { w[$0].floatValue }
    }
}
