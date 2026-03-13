//
//  Date+Extensions.swift
//  ios-app
//
//  Created by Ridhwik Kalgaonkar on 11/8/24.
//

import Foundation

extension Date {
    func toString() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd-HH-mm-ss"
        return formatter.string(from: self)
    }
}
