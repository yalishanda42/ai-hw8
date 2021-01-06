// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NeuralNetwork",
    products: [
        .library(
            name: "NeuralNetwork",
            targets: ["NeuralNetwork"]),
    ],
    dependencies: [
         .package(url: "https://github.com/apple/swift-numerics", from: "0.0.8"),
    ],
    targets: [
        .target(
            name: "NeuralNetwork",
            dependencies: [
                .product(name: "Numerics", package: "swift-numerics"),
            ]),
        .testTarget(
            name: "NeuralNetworkTests",
            dependencies: ["NeuralNetwork"]),
    ]
)
