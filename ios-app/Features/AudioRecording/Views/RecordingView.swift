//
//  RecordingView.swift
//  ios-app
//
//  Created by Ridhwik Kalgaonkar on 11/8/24.
//

import SwiftUI
import SiriWaveView
import AVFoundation

struct RecordingView: View {
    @StateObject private var recorder = AudioRecorderViewModel()
    @State private var apiResponse: String = ""
    @State private var isLoading: Bool = false
    @State private var isImporting = false
    @State private var importError: String?
    @State private var uploadError: Error?
    @State private var showUploadError = false
    @State private var showDeleteAlert = false
    @State private var recordingToDelete: URL?
    
    var body: some View {
        NavigationView {
            VStack {
                // Siri Waveform visualization
                SiriWaveView(power: $recorder.currentPower)
                    .opacity(0.5)
                    .frame(height: 100)
                    .padding()
                    .background(Color.black.opacity(0.1))
                    .cornerRadius(10)
                
                // Audio Player Controls
                if recorder.currentlyPlayingURL != nil {
                    AudioPlayerControlsView(recorder: recorder)
                        .padding(.vertical)
                }
                
                HStack(spacing: 20) {
                    Button(action: {
                        if recorder.isRecording {
                            recorder.stopRecording()
                        } else {
                            recorder.startRecording()
                        }
                    }) {
                        Image(systemName: recorder.isRecording ? "stop.circle.fill" : "mic.circle.fill")
                            .font(.system(size: 64))
                            .foregroundColor(recorder.isRecording ? .red : .blue)
                    }
                    
                    Button(action: {
                        isImporting = true
                    }) {
                        Image(systemName: "square.and.arrow.down")
                            .font(.system(size: 64))
                            .foregroundColor(.blue)
                    }
                    .fileImporter(
                        isPresented: $isImporting,
                        allowedContentTypes: [.audio],
                        allowsMultipleSelection: false
                    ) { result in
                        Task {
                            do {
                                let url = try result.get().first!
                                print("Selected file URL: \(url)")
                                await importAudio(from: url)
                            } catch {
                                print("File import error: \(error)")
                                importError = error.localizedDescription
                            }
                        }
                    }
                }
                
                VStack {
                    Button(action: {
                        Task {
                            await fetchHelloMessage()
                        }
                    }) {
                        HStack {
                            Text("Test API")
                            if isLoading {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle())
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .disabled(isLoading)
                    
                    if !apiResponse.isEmpty {
                        Text(apiResponse)
                            .padding()
                            .background(Color.gray.opacity(0.1))
                            .cornerRadius(8)
                    }
                    
                    if let error = importError {
                        Text(error)
                            .foregroundColor(.red)
                            .padding()
                    }
                }
                .padding()
                
                List {
                    ForEach(recorder.recordings, id: \.self) { recording in
                        HStack {
                            Group {
                                if recorder.currentlyPlayingURL == recording {
                                    Button(action: {
                                        if recorder.isPlaying {
                                            recorder.pausePlayback()
                                        } else {
                                            recorder.resumePlayback()
                                        }
                                    }) {
                                        Image(systemName: recorder.isPlaying ? "pause.circle.fill" : "play.circle.fill")
                                            .foregroundColor(.red)
                                    }
                                    .buttonStyle(BorderlessButtonStyle())
                                } else {
                                    Button(action: {
                                        recorder.startPlayback(audio: recording)
                                    }) {
                                        Image(systemName: "play.circle.fill")
                                            .foregroundColor(.blue)
                                    }
                                    .buttonStyle(BorderlessButtonStyle())
                                }
                                
                                Text(recording.lastPathComponent)
                                    .foregroundColor(.primary)
                            }
                            
                            Spacer()
                            
                            HStack(spacing: 16) {
                                if recorder.uploadingFiles.contains(recording) {
                                    ProgressView()
                                } else {
                                    Button(action: {
                                        Task {
                                            do {
                                                try await recorder.uploadAudio(from: recording)
                                            } catch {
                                                uploadError = error
                                                showUploadError = true
                                            }
                                        }
                                    }) {
                                        Image(systemName: "icloud.and.arrow.up")
                                            .foregroundColor(.blue)
                                    }
                                    .buttonStyle(BorderlessButtonStyle())
                                }
                                
                                Button(action: {
                                    recordingToDelete = recording
                                    showDeleteAlert = true
                                }) {
                                    Image(systemName: "trash")
                                        .foregroundColor(.red)
                                }
                                .buttonStyle(BorderlessButtonStyle())
                            }
                        }
                        .contentShape(Rectangle())
                    }
                    .listRowBackground(Color.clear)
                }
                .listStyle(PlainListStyle())
            }
            .navigationTitle("Voice Recorder")
            .alert("Upload Error", isPresented: $showUploadError, presenting: uploadError) { _ in
                Button("OK", role: .cancel) {}
            } message: { error in
                Text(error.localizedDescription)
            }
            .alert("Delete Recording", isPresented: $showDeleteAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Delete", role: .destructive) {
                    if let url = recordingToDelete {
                        recorder.deleteRecording(url: url)
                    }
                }
            } message: {
                Text("Are you sure you want to delete this recording?")
            }
            .alert("Upload Successful", isPresented: $recorder.showUploadSuccess) {
                Button("OK", role: .cancel) {}
            } message: {
                Text(recorder.uploadResponse)
            }
        }
    }
    
    private func importAudio(from url: URL) async {
        do {
            try await recorder.importAudio(from: url)
            print("Audio import successful")
        } catch {
            print("Audio import failed: \(error)")
            importError = "Failed to import audio: \(error.localizedDescription)"
        }
    }
    
    func fetchHelloMessage() async {
        isLoading = true
        defer { isLoading = false }
        
        guard let url = URL(string: "http://127.0.0.1:5000/api/hello") else {
            apiResponse = "Invalid URL"
            return
        }
        
        do {
            let (data, _) = try await URLSession.shared.data(from: url)
            
            if let response = String(data: data, encoding: .utf8) {
                await MainActor.run {
                    apiResponse = response
                }
            }
        } catch {
            await MainActor.run {
                apiResponse = "Error: \(error.localizedDescription)"
            }
        }
    }
}
