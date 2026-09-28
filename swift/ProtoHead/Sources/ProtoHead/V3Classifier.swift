import Foundation

/// constants from runs/fusion/v3_constants.json (fusion.py --export)
public struct V3Constants: Codable {
    public let head_weight: [[Float]]   // 5 x 68, CNN's last linear layer
    public let head_bias: [Float]
    public let rr_mean: [Float]         // timing model: standardize the 4 normalized RR features
    public let rr_scale: [Float]
    public let rr_coef: [[Float]]       // 4 x 4, classes N S V F
    public let rr_intercept: [Float]
    public let w: Float
    public let classes: [String]

    public static func load(_ url: URL) throws -> V3Constants {
        try JSONDecoder().decode(V3Constants.self, from: Data(contentsOf: url))
    }
}

/// v3 = CNN + timing model. Same as fuse(mode="multi") in fusion.py:
///   score = log_softmax(CNN logits) + w * log(timing model probs), Q gets the mean of the 4
public struct V3Classifier {
    public let c: V3Constants

    public init(constants: V3Constants) { c = constants }

    static func logSoftmax(_ z: [Float]) -> [Float] {
        let m = z.max()!
        let lse = m + log(z.reduce(0) { $0 + exp($1 - m) })
        return z.map { $0 - lse }
    }

    /// embedding: 64 from Core ML, rr: the 4 raw RR features, rrNorm: the 4 normalized ones
    public func scores(embedding: [Float], rr: [Float], rrNorm: [Float]) -> [Float] {
        let f = embedding + rr
        let logits = (0..<c.head_weight.count).map { k in
            zip(c.head_weight[k], f).reduce(c.head_bias[k]) { $0 + $1.0 * $1.1 }
        }
        var s = V3Classifier.logSoftmax(logits)

        let z = (0..<rrNorm.count).map { (rrNorm[$0] - c.rr_mean[$0]) / c.rr_scale[$0] }
        let dec = (0..<c.rr_coef.count).map { k in
            zip(c.rr_coef[k], z).reduce(c.rr_intercept[k]) { $0 + $1.0 * $1.1 }
        }
        // python clips probabilities at 1e-9 before the log, do the same
        let lp = V3Classifier.logSoftmax(dec).map { max($0, log(Float(1e-9))) }
        for k in 0..<lp.count { s[k] += c.w * lp[k] }
        s[4] += c.w * lp.reduce(0, +) / Float(lp.count)
        return s
    }

    public func classify(embedding: [Float], rr: [Float], rrNorm: [Float]) -> Int {
        let s = scores(embedding: embedding, rr: rr, rrNorm: rrNorm)
        return s.indices.max { s[$0] < s[$1] }!
    }
}
