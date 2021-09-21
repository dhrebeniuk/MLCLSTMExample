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
    
    var lstmInputWeightTensors = [MLCTensor]()
    var lstmInputBiasTensors = [MLCTensor]()
    var lstmHiddenWeightTensors = [MLCTensor]()

    var lossLabelTensor: MLCTensor!

    var lstmTensor: MLCTensor!
    
    var denseOutput: MLCTensor!
    var output: MLCTensor!

    let batchSize = 1
    let windowSize = 1
    let sampleSize = 1

    let lstmLayersCount = 4

    let epochs = 90
    
    
    let futureOffset = 50
    let useStates = 1
    
    func setup() {
        initializeTensors()
        buildGraph()
        buildTrainingGraph()
        buildInferenceGraph()
    }
}

extension LSTMTrainEngine {
    
    private func initializeTensors() {
        device = MLCDevice(type: .cpu)!

        inputTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [batchSize, windowSize, sampleSize], dataType: .float32)!)

        lossLabelTensor = MLCTensor(descriptor: MLCTensorDescriptor(shape: [batchSize, windowSize, sampleSize], dataType: .float32)!)
        
        lstmInputWeightTensors = [MLCTensor](repeating: MLCTensor(descriptor: MLCTensorDescriptor(shape: [batchSize, sampleSize*sampleSize], dataType: .float32)!, randomInitializerType: .xavier), count: 4*lstmLayersCount)
        
        lstmInputBiasTensors = [MLCTensor](repeating: MLCTensor(descriptor: MLCTensorDescriptor(shape:  [batchSize, sampleSize], dataType: .float32)!, randomInitializerType: .xavier), count: 4 * lstmLayersCount)
        
        lstmHiddenWeightTensors = [MLCTensor](repeating: MLCTensor(descriptor: MLCTensorDescriptor(shape:  [batchSize, sampleSize*sampleSize], dataType: .float32)!, randomInitializerType: .xavier), count: 4 * lstmLayersCount)
    }
    
    private func buildGraph() {
        graph = MLCGraph()
        
        let lstmDescriptor = MLCLSTMDescriptor(inputSize: sampleSize,       // Sample size for input data
                                               hiddenSize: sampleSize,      // Sample size for output data
                                               layerCount: lstmLayersCount, // LSTM layers count
                                               usesBiases: true,            // Use biases
                                               batchFirst: true,            // Batch size as first paramter: [batchSize, windowSize, sampleSize]
                                               isBidirectional: false,      // We using sigle direction for this task
                                               returnsSequences: true,      // Need return sequences
                                               dropout: 0.0,                // In IOS 14 LSTM doen't support dropout in MLCompute
                                               resultMode: .outputAndStates)// Setup return results and states
        
        let lstmLayer = MLCLSTMLayer(descriptor: lstmDescriptor,
                                     inputWeights: self.lstmInputWeightTensors, // Need setup 4 weight tensors for every LSTM Layer
                                     hiddenWeights: self.lstmHiddenWeightTensors, // Need setup 4 hidden tensors for every LSTM Layer
                                     biases: self.lstmInputBiasTensors)  // Need setup 4 weight tensors for every LSTM Layer

        lstmTensor = graph.node(with: lstmLayer!, sources: [inputTensor])
    }
    
    private func buildTrainingGraph() {
        let optimizerDescriptor = MLCOptimizerDescriptor(learningRate: 1e-5,
                                                         gradientRescale: 1.0,
                                                         regularizationType: .l2 ,
                                                         regularizationScale: 0.1)

        trainingGraph = MLCTrainingGraph(graphObjects: [graph],
                                         lossLayer: MLCLossLayer(descriptor: MLCLossDescriptor(type: .huber, reductionType: .mean)),
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

    func execTrainingLoop(trainingData: [Float], resultHandlerLog: @escaping (([Float], Int) -> Void)) {
        _ = self.batchSize
        let windowSize = self.windowSize
        _ = self.sampleSize

        let lstmLayersCount = self.lstmLayersCount

        for epoch in 0..<epochs {
            var resultArray: [Float] = [Float]()

            var hiddenStates = [[Float]](repeating: [Float](repeating: 0.0, count: windowSize), count: lstmLayersCount)
            var cellStates = [[Float]](repeating: [Float](repeating: 0.0, count: windowSize), count: lstmLayersCount)
                        
            for windowIndex in 0..<(trainingData.count/windowSize) - futureOffset {
                var window = Array(trainingData[windowIndex*windowSize..<(windowIndex+1)*windowSize])
                let nextWindow = Array(trainingData[(windowIndex + futureOffset)*windowSize..<(windowIndex + futureOffset + 1)*windowSize])

                for layerIndex in 0..<lstmLayersCount {
                    window.append(contentsOf: hiddenStates[layerIndex])
                    window.append(contentsOf: cellStates[layerIndex])
                }
                
                let windowData = window.withUnsafeBufferPointer { pointer in
                    MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                                  length: windowSize * MemoryLayout<Float>.size * (1 + useStates * 2 * lstmLayersCount))
                }

                let nextWindowData = nextWindow.withUnsafeBufferPointer { pointer in
                    MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                                  length: windowSize*MemoryLayout<Float>.size)
                }

                self.trainingGraph.execute(inputsData: ["data" : windowData],
                                      lossLabelsData: ["prediction" : nextWindowData],
                                      lossLabelWeightsData: nil,
                                      batchSize: batchSize,
                                      options: [.synchronous]) { [self] (r, e, time) in
                    
                    let bufferOutput = UnsafeMutableRawPointer.allocate(byteCount: windowSize * MemoryLayout<Float>.size * (1 + useStates * lstmLayersCount * 2), alignment: MemoryLayout<Float>.alignment)
                    
                    lstmTensor?.copyDataFromDeviceMemory(toBytes: bufferOutput, length: windowSize * MemoryLayout<Float>.size * (1 + useStates * lstmLayersCount * 2), synchronizeWithDevice: false)
                    
                    let float4Ptr = bufferOutput.bindMemory(to: Float.self, capacity: windowSize * (1 + useStates * lstmLayersCount * 2))
                    let float4Buffer = UnsafeBufferPointer(start: float4Ptr, count: windowSize * (1 + useStates * lstmLayersCount * 2))
                    let predictedArray = Array(float4Buffer[0..<windowSize])
                    
                    for layerIndex in 0..<lstmLayersCount {
                        let hiddenState = Array(float4Buffer[windowSize*(1 + layerIndex * 2)..<windowSize*(2 + layerIndex * 2)])
                        let cellState = Array(float4Buffer[windowSize*(2 + layerIndex * 2)..<windowSize*(3 + layerIndex * 2)])

                        hiddenStates[layerIndex] = hiddenState
                        cellStates[layerIndex] = cellState
                    }

                    resultArray = predictedArray
                }
            }
            
            resultHandlerLog(resultArray, epoch)
        }
    }

    func predictForecastGraph(evaluteData: [Float], resultHandler: @escaping (([Float]) -> Void)) {
        var resultArray: [Float] = [Float]()

        _ = self.batchSize
        let windowSize = self.windowSize
        _ = self.sampleSize

        let lstmLayersCount = self.lstmLayersCount
        
        var hiddenStates = [[Float]](repeating: [Float](repeating: 0.0, count: windowSize), count: lstmLayersCount)
        var cellStates = [[Float]](repeating: [Float](repeating: 0.0, count: windowSize), count: lstmLayersCount)
        
        for windowIndex in 0..<(evaluteData.count/windowSize)-1 {
            var window = Array(evaluteData[windowIndex * windowSize..<(windowIndex + 1) * windowSize])

            for layerIndex in 0..<lstmLayersCount {
                window.append(contentsOf: hiddenStates[layerIndex])
                window.append(contentsOf: cellStates[layerIndex])
            }

            let windowData = window.withUnsafeBufferPointer { pointer in
                MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                              length: windowSize * MemoryLayout<Float>.size * (1 + useStates * 2 * lstmLayersCount))
            }
            
            self.inferenceGraph.execute(inputsData: ["data" : windowData],
                                  batchSize: batchSize,
                                  options: [.synchronous]) { [self] (r, e, time) in
             
                let bufferOutput = UnsafeMutableRawPointer.allocate(byteCount: windowSize * MemoryLayout<Float>.size * (1 + useStates * lstmLayersCount * 2), alignment: MemoryLayout<Float>.alignment)
                
                lstmTensor?.copyDataFromDeviceMemory(toBytes: bufferOutput, length: windowSize * MemoryLayout<Float>.size * (1 + useStates * lstmLayersCount * 2), synchronizeWithDevice: false)
                
                let float4Ptr = bufferOutput.bindMemory(to: Float.self, capacity: windowSize * (1 + useStates * lstmLayersCount * 2))
                let float4Buffer = UnsafeBufferPointer(start: float4Ptr, count: windowSize * (1 + useStates * lstmLayersCount * 2))
                let predictedArray = Array(float4Buffer[0..<windowSize])
                
                for layerIndex in 0..<lstmLayersCount {
                    let hiddenState = Array(float4Buffer[windowSize*(1 + layerIndex * 2)..<windowSize*(2 + layerIndex * 2)])
                    let cellState = Array(float4Buffer[windowSize*(2 + layerIndex * 2)..<windowSize*(3 + layerIndex * 2)])

                    hiddenStates[layerIndex] = hiddenState
                    cellStates[layerIndex] = cellState
                }

                resultArray.append(contentsOf: predictedArray)
            }
        }
        
        resultHandler(resultArray)
    }
}
