//
//  LSTMTrainEngine.swift
//  MLCLSTMExample
//
//  Created by Hrebeniuk Dmytro on 20.09.2021.
//

import Foundation
import MLCompute


class LSTMTrainEngine {
    
    var device: MLCDevice!
    var graph: MLCGraph!
    
    var trainingGraph: MLCTrainingGraph!
    var inferenceGraph: MLCInferenceGraph!
    var inputTensor: MLCTensor!
    
    var lstmTensor: MLCTensor!
    var lstmInputWeightTensors = [MLCTensor]()
    var lstmInputBiasTensors = [MLCTensor]()
    var lstmHiddenWeightTensors = [MLCTensor]()

    var lstm2Tensor: MLCTensor!
    var lstm2InputWeightTensors = [MLCTensor]()
    var lstm2InputBiasTensors = [MLCTensor]()
    var lstm2HiddenWeightTensors = [MLCTensor]()
    
    var lossLabelTensor: MLCTensor!

    
    var denseOutput: MLCTensor!
    var denseOutputWeightTensors: MLCTensor!
    var denseOutputBiasTensors: MLCTensor!
    
    var softMax: MLCTensor!

    var output: MLCTensor!

    let batchSize = 1
    let windowSize = 100
    let sampleSize = 1

    let lstmLayersCount = 1

    let epochs = 10000

    let hiddenSize = 512
    
    let categories = 3
    let futureOffset = 0
    
    func setup() {
        initializeTensors()
        buildGraph()
        buildTrainingGraph()
        buildInferenceGraph()
    }
}

extension LSTMTrainEngine {
    
    private func initializeTensors() {
        device = MLCDevice(type: .any)!

        inputTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, 1, sampleSize, 1], dataType: .float32)!)

        lossLabelTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, categories], dataType: .float32)!)
        
        lstmInputWeightTensors = [MLCTensor](repeating: MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, sampleSize * hiddenSize, 1, 1], dataType: .float32)!, randomInitializerType: .xavier), count: 4 * lstmLayersCount)

        lstmInputBiasTensors = [MLCTensor](repeating: MLCTensor(descriptor: MLCTensorDescriptor(shape:  [1, hiddenSize, 1, 1], dataType: .float32)!, randomInitializerType: .xavier), count: 4 * lstmLayersCount)

        lstmHiddenWeightTensors = [MLCTensor](repeating: MLCTensor(descriptor: MLCTensorDescriptor(shape:  [1, hiddenSize * hiddenSize, 1, 1], dataType: .float32)!, randomInitializerType: .xavier), count: 4 * lstmLayersCount)
        
        lstm2InputWeightTensors = [MLCTensor](repeating: MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, hiddenSize * hiddenSize, 1, 1], dataType: .float32)!, randomInitializerType: .xavier), count: 4*lstmLayersCount)

        lstm2InputBiasTensors = [MLCTensor](repeating: MLCTensor(descriptor: MLCTensorDescriptor(shape:  [1, hiddenSize, 1, 1], dataType: .float32)!, randomInitializerType: .xavier), count: 4 * lstmLayersCount)

        lstm2HiddenWeightTensors = [MLCTensor](repeating: MLCTensor(descriptor: MLCTensorDescriptor(shape:  [1, hiddenSize * hiddenSize, 1, 1], dataType: .float32)!, randomInitializerType: .xavier), count: 4 * lstmLayersCount)

        denseOutputWeightTensors = MLCTensor(descriptor: MLCTensorDescriptor(shape: [1, hiddenSize * categories, 1, 1], dataType: .float32)!, randomInitializerType: .xavier)
        denseOutputBiasTensors = MLCTensor(descriptor: MLCTensorDescriptor(shape: [1,  categories, 1, 1], dataType: .float32)!, randomInitializerType: .xavier)
    }
    
    private func buildGraph() {
        graph = MLCGraph()
        
        let lstmDescriptor = MLCLSTMDescriptor(inputSize: sampleSize,       // Sample size for input data
                                               hiddenSize: hiddenSize,      // Sample size for output data
                                               layerCount: lstmLayersCount, // LSTM layers count
                                               usesBiases: true,            // Use biases
                                               batchFirst: true,            // Batch size as first paramter: [batchSize, windowSize, sampleSize]
                                               isBidirectional: false,      // We using sigle direction for this task
                                               returnsSequences: true,      // Need return sequences
                                               dropout: 0.0,                // In IOS 14 LSTM doen't support dropout in MLCompute
                                               resultMode: .output)// Setup return results and states

        let lstm2Descriptor = MLCLSTMDescriptor(inputSize: hiddenSize,       // Sample size for input data
                                               hiddenSize: hiddenSize,      // Sample size for output data
                                               layerCount: lstmLayersCount, // LSTM layers count
                                               usesBiases: true,            // Use biases
                                               batchFirst: true,            // Batch size as first paramter: [batchSize, windowSize, sampleSize]
                                               isBidirectional: false,      // We using sigle direction for this task
                                               returnsSequences: true,      // Need return sequences
                                               dropout: 0.0,                // In IOS 14 LSTM doen't support dropout in MLCompute
                                               resultMode: .output)// Setup return results and states

        
        
        let lstmLayer = MLCLSTMLayer(descriptor: lstmDescriptor,
                                     inputWeights: self.lstmInputWeightTensors, // Need setup 4 weight tensors for every LSTM Layer
                                     hiddenWeights: self.lstmHiddenWeightTensors, // Need setup 4 hidden tensors for every LSTM Layer
                                     biases: self.lstmInputBiasTensors)  // Need setup 4 weight tensors for every LSTM Layer
        lstmTensor = graph.node(with: lstmLayer!, sources: [inputTensor])

        let lstm2Layer = MLCLSTMLayer(descriptor: lstm2Descriptor,
                                     inputWeights: self.lstm2InputWeightTensors, // Need setup 4 weight tensors for every LSTM Layer
                                     hiddenWeights: self.lstm2HiddenWeightTensors, // Need setup 4 hidden tensors for every LSTM Layer
                                     biases: self.lstm2InputBiasTensors)  // Need setup 4 weight tensors for every LSTM Layer


        lstm2Tensor = graph.node(with: lstm2Layer!, sources: [lstmTensor])

        let denseOutputDescriptor = MLCConvolutionDescriptor(kernelSizes: (height: hiddenSize/* input */, width: categories /* ouput */),
                                                             inputFeatureChannelCount: hiddenSize,
                                                             outputFeatureChannelCount: categories)

        let denseOutputLayer = MLCFullyConnectedLayer(weights: denseOutputWeightTensors,
                                                      biases: denseOutputBiasTensors,
                                                      descriptor: denseOutputDescriptor)

        denseOutput = graph.node(with: denseOutputLayer!, sources: [lstm2Tensor])

        //        sigmoid = graph.node(with: MLCActivationLayer(descriptor: MLCActivationDescriptor(type: MLCActivationType.sigmoid)!), source: lstmTensor!)

        softMax = graph.node(with: MLCSoftmaxLayer(operation: .softmax),
                             source: denseOutput!)
    }
    
    private func buildTrainingGraph() {
        let optimizerDescriptor = MLCOptimizerDescriptor(learningRate: 1e-4,
                                                         gradientRescale: 1.0,
                                                         regularizationType: .none,
                                                         regularizationScale: 0.1)

        trainingGraph = MLCTrainingGraph(graphObjects: [graph],
                                         lossLayer: MLCLossLayer(descriptor: MLCLossDescriptor(type: .softmaxCrossEntropy, reductionType: .mean)),
            optimizer: MLCAdamOptimizer(descriptor: optimizerDescriptor, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, timeStep: 1))

        trainingGraph.addInputs(["data" : inputTensor],
                                lossLabels: ["prediction" : lossLabelTensor])

        trainingGraph.compile(options: [], device: device)
    }
    
    private func buildInferenceGraph() {
        inferenceGraph = MLCInferenceGraph(graphObjects: [graph])
        inferenceGraph.addInputs(["data" : inputTensor])
        inferenceGraph.compile(options: [], device: device)
    }
    
}

extension LSTMTrainEngine {

    func execTrainingLoop(trainingData: [Float], yData: [Int], resultHandlerLog: @escaping ((Int) -> Void)) {
        _ = self.batchSize
        let windowSize = self.windowSize
        _ = self.sampleSize

        for epoch in 0..<epochs {
            for windowIndex in 0..<(trainingData.count/windowSize) {
                let x = Array(trainingData[windowIndex * windowSize..<(windowIndex + 1)*windowSize])
                let y = Array(yData[windowIndex * windowSize..<(windowIndex + 1) * windowSize])
                    .flatMap { self.oneHotEncoding($0, length: self.categories) }

                let windowData = x.withUnsafeBufferPointer { pointer in
                    MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                                  length: windowSize * MemoryLayout<Float>.size)
                }

                let yWindowData = y.withUnsafeBufferPointer { pointer in
                    MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                                  length: windowSize * MemoryLayout<Float>.size * categories)
                }

                self.trainingGraph.execute(inputsData: ["data" : windowData],
                                      lossLabelsData: ["prediction" : yWindowData],
                                      lossLabelWeightsData: nil,
                                      batchSize: batchSize,
                                      options: [.synchronous]) { (r, e, time) in
                    
                    let windowSize = 1
                    let bufferOutput = UnsafeMutableRawPointer.allocate(byteCount: windowSize * MemoryLayout<Float>.size , alignment: MemoryLayout<Float>.alignment)
                    
                    r?.copyDataFromDeviceMemory(toBytes: bufferOutput, length: windowSize * MemoryLayout<Float>.size, synchronizeWithDevice: false)

                    let float4Ptr = bufferOutput.bindMemory(to: Float.self, capacity: windowSize)
                    let float4Buffer = UnsafeBufferPointer(start: float4Ptr, count: windowSize)
                    let predictedArray = Array(float4Buffer[0..<windowSize])
                    print(predictedArray)
                }
            }
            
            resultHandlerLog(epoch)
        }
    }

    func predictForecastGraph(evaluteData: [Float], resultHandler: @escaping (([Float]) -> Void)) {
        var resultArray: [Float] = [Float]()

        _ = self.batchSize
        let windowSize = self.windowSize
        _ = self.sampleSize

        for windowIndex in 0..<(evaluteData.count/windowSize)-1 {
            let window = Array(evaluteData[windowIndex * windowSize..<(windowIndex + 1) * windowSize])

            let windowData = window.withUnsafeBufferPointer { pointer in
                MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                              length: windowSize * MemoryLayout<Float>.size)
            }
            
            self.inferenceGraph.execute(inputsData: ["data" : windowData],
                                  batchSize: batchSize,
                                  options: [.synchronous]) { [self] (r, e, time) in
             
                let windowSize = 1
                let bufferOutput = UnsafeMutableRawPointer.allocate(byteCount: windowSize * MemoryLayout<Float>.size * categories, alignment: MemoryLayout<Float>.alignment)
                
                r?.copyDataFromDeviceMemory(toBytes: bufferOutput, length: windowSize * MemoryLayout<Float>.size * categories, synchronizeWithDevice: false)
                
                let float4Ptr = bufferOutput.bindMemory(to: Float.self, capacity: windowSize * categories)
                let float4Buffer = UnsafeBufferPointer(start: float4Ptr, count: windowSize * categories)
                let predictedArray = Array(float4Buffer[0..<windowSize * categories])
                
                let decoded = self.argmaxDecoding(predictedArray)
                if decoded.isNaN == false {
                    resultArray.append(contentsOf: [Float].init(repeating: decoded, count: self.windowSize))
                }
            }
        }
        
        resultHandler(resultArray)
    }
    
    private func oneHotEncodingInt(_ number: Int, length: Int = 10) -> [Int] {
        guard number < length else {
            fatalError("wrong ordinal vs encoding length")
        }
        
        var array = Array<Int>(repeating: 0, count: length)
        array[number] = 1
        return array
    }
    
    private func oneHotEncoding(_ number: Int, length: Int = 10) -> [Float] {
        guard number < length else {
            fatalError("wrong ordinal vs encoding length")
        }
        
        var array = Array<Float>(repeating: 0.0, count: length)
        array[number] = 1.0
        return array
    }
    
    private func oneHotDecoding(_ encoding: [Float]) -> Int {
        var value: Int = 0
        
        for i in 0..<encoding.count {
            if encoding[i] == 1 {
                value = i
                break
            }
        }
        
        return value
    }
    
    private func argmaxDecoding(_ encoding: [Float]) -> Float {
        var max: Float = 0
        var pos: Int = 0
        
        for i in 0..<encoding.count {
            if encoding[i] > max {
                max = encoding[i]
                pos = i
            }
        }
        
        return Float(pos)
    }
}

