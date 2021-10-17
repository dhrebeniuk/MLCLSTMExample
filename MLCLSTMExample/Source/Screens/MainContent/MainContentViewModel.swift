//
//  MainContentViewModel.swift
//  MLCLSTMExample
//
//  Created by Hrebeniuk Dmytro on 20.09.2021.
//

import Foundation
import Combine
import SwiftUI


class MainContentViewModel: ObservableObject {
    
    @Published var generatedData: [Double] = [Double]()
    @Published var predictedData: [Double] = [Double]()
    @Published var epochIndex: Int = 0

    let lstmTrainEngine = LSTMTrainEngine()
    
    func setup () {
        var xData = [Double]()
        var yData = [Int]()
        
        for index in 0..<200 {
            let a = Double(index) / 10.0
            let randomValue = (Double(arc4random()%100) / 50.0)
            let value = pow(sin(a), 2.0) + abs(randomValue)
            xData.append(value)
            yData.append(0)
        }
        
        for index in 0..<200 {
            let a = Double(index) / 10.0
            let randomValue = (Double(arc4random()%100) / 50.0)
            let value = pow(sin(a), 5.0) + abs(randomValue)
            xData.append(value)
            yData.append(1)
        }

        for index in 0..<200 {
            let a = Double(index) / 10.0
            let randomValue = (Double(arc4random()%100) / 50.0)
            let value = pow(sin(a)*cos(a), 2.0) + abs(randomValue)
            xData.append(value)
            yData.append(2)
        }
        
        for index in 0..<100 {
            let a = Double(index) / 10.0
            let randomValue = (Double(arc4random()%100) / 50.0)
            let value = pow(sin(a), 5.0) + abs(randomValue)
            xData.append(value)
            yData.append(1)
        }
        
        for index in 0..<100 {
            let a = Double(index) / 10.0
            let randomValue = (Double(arc4random()%100) / 50.0)
            let value = pow(sin(a), 2.0) + abs(randomValue)
            xData.append(value)
            yData.append(0)
        }
    
        generatedData.append(contentsOf: xData)
        
        lstmTrainEngine.setup()
        
        DispatchQueue.global().async {
            let floatData = self.generatedData.map { Float($0) }
            self.lstmTrainEngine.execTrainingLoop(trainingData: floatData, yData: yData) { epochIndex in
                if epochIndex%5 == 0 {
                    self.lstmTrainEngine.predictForecastGraph(evaluteData: floatData) { predictedData in
                        DispatchQueue.main.async {
                            self.epochIndex = epochIndex + 1
                            self.predictedData = predictedData.map { Double($0) }
                        }
                    }
                }
                else {
                    DispatchQueue.main.async {
                        self.epochIndex = epochIndex + 1
                    }
                }

            }
        }
    }
    
    
}
