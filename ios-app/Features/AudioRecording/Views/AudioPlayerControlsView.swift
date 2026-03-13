//
//  AudioPlayerControlsView.swift
//  ios-app
//
//  Created by Ridhwik Kalgaonkar on 11/22/24.
//

import SwiftUI

struct AudioPlayerControlsView: View {
    @ObservedObject var recorder: AudioRecorderViewModel
    
    var body: some View {
        VStack(spacing: 8) {
            if recorder.currentlyPlayingURL != nil {
                Text(recorder.currentlyPlayingURL?.lastPathComponent ?? "")
                    .font(.caption)
                    .lineLimit(1)
                
                HStack {
                    Text(timeString(from: recorder.currentTime))
                        .font(.caption)
                        .monospacedDigit()
                    
                    Slider(
                        value: Binding(
                            get: { recorder.currentTime },
                            set: { recorder.seek(to: $0) }
                        ),
                        in: 0...max(recorder.duration, 1)
                    )
                    .disabled(recorder.currentlyPlayingURL == nil)
                    
                    Text(timeString(from: recorder.duration))
                        .font(.caption)
                        .monospacedDigit()
                }
            }
        }
        .padding(.horizontal)
    }
    
    private func timeString(from timeInterval: TimeInterval) -> String {
        let minutes = Int(timeInterval / 60)
        let seconds = Int(timeInterval.truncatingRemainder(dividingBy: 60))
        return String(format: "%02d:%02d", minutes, seconds)
    }
}
