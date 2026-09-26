// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "ProtoHead",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [.library(name: "ProtoHead", targets: ["ProtoHead"])],
    targets: [
        .target(name: "ProtoHead"),
        // mac only tool that replays a MIT-BIH record exported by replay.py
        .executableTarget(name: "replay", dependencies: ["ProtoHead"]),
        // mac only, checks MLUpdateTask fine-tune against python
        .executableTarget(name: "finetune", dependencies: ["ProtoHead"]),
    ]
)
