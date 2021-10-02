//
//  KalmanFilter.swift
//  MLCLSTMExample
//
//  Created by Hrebeniuk Dmytro on 27.09.2021.
//

import Foundation


class KalmanFilter<T: FloatingPoint> {
    
    let varianceProcess: T

    init(varianceProcess: T = T(1)) {
        self.varianceProcess = varianceProcess
    }
    
    private(set) var variance = T(0)
    private(set) var Pc = T(0)
    private(set) var G = T(0)
    private(set) var P = T(1)
    private(set) var Xe = T(0)
    
    func fit(values: [T]) -> [T] {
        let mean = values.reduce(T(0), { $0 + $1 }) / T(values.count)

        let squareSumm = (values.reduce(T(0), {$0 + (mean - $1) * (mean - $1) } ))
        let dispersion = squareSumm / (T(values.count))
        variance = sqrt(dispersion)
        
        Pc = T(0)
        G = T(0)
        P = T(1)
        Xe = T(0)
        
        func filter(value: T) -> T {
            Pc = P + varianceProcess
            G = Pc / (Pc + variance)
            P = (T(1) - G) * Pc
            
            let Xp = Xe
            let Zp = Xp
            Xe = G * (value - Zp) + Xp
            return Xe
        }
        
        return values.map { filter(value: $0) }
    }
    
}
