//
//  AudioRecorderViewModel.swift
//  ios-app
//
//  Created by Ridhwik Kalgaonkar on 11/8/24.
//

import AVFoundation
import SwiftUI

class AudioRecorderViewModel: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var recordings: [URL] = []
    @Published var currentPower: Double = 0.0
    @Published var uploadingFiles: Set<URL> = []
    @Published var currentlyPlayingURL: URL?
    @Published var isPlaying = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0
    @Published var uploadResponse: String = ""
    @Published var showUploadSuccess = false
    
    var audioRecorder: AVAudioRecorder?
    var audioPlayer: AVAudioPlayer?
    private var timer: Timer?
    private var playbackTimer: Timer?
    
    override init() {
        super.init()
        fetchRecordings()
    }
    
    func deleteRecording(url: URL) {
        do {
            // If the file is currently playing, stop it first
            if currentlyPlayingURL == url {
                stopPlayback()
            }
            
            try FileManager.default.removeItem(at: url)
            // Refresh the recordings list
            fetchRecordings()
        } catch {
            print("Error deleting recording: \(error.localizedDescription)")
        }
    }
    
    func importAudio(from sourceURL: URL) async throws {
        let documentPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let destinationURL = documentPath.appendingPathComponent(sourceURL.lastPathComponent)
        
        print("Attempting to import from: \(sourceURL)")
        print("Importing to: \(destinationURL)")
        
        if sourceURL.startAccessingSecurityScopedResource() {
            defer { sourceURL.stopAccessingSecurityScopedResource() }
            
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                print("File already exists, removing old version")
                try FileManager.default.removeItem(at: destinationURL)
            }
            
            try FileManager.default.copyItem(at: sourceURL, to: destinationURL)
            print("File successfully copied to app directory")
            
            DispatchQueue.main.async {
                self.fetchRecordings()
                print("Recordings list updated")
                print("Current recordings: \(self.recordings)")
            }
        }
    }
    
    func uploadAudio(from url: URL) async throws {
        guard let audioData = try? Data(contentsOf: url) else {
            throw NSError(domain: "AudioUpload", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not read audio file"])
        }
        
        let boundary = UUID().uuidString
        var request = URLRequest(url: URL(string: "http://127.0.0.1:5000/api/upload-audio")!)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"audio\"; filename=\"\(url.lastPathComponent)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/m4a\r\n\r\n".data(using: .utf8)!)
        body.append(audioData)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        DispatchQueue.main.async {
            self.uploadingFiles.insert(url)
        }
        
        do {
            defer {
                DispatchQueue.main.async {
                    self.uploadingFiles.remove(url)
                }
            }
            
            let (data, response) = try await URLSession.shared.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw NSError(domain: "AudioUpload", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid response"])
            }
            
            if httpResponse.statusCode == 200 {
                let responseString = String(data: data, encoding: .utf8) ?? ""
                DispatchQueue.main.async {
                    self.uploadResponse = responseString
                    self.showUploadSuccess = true
                }
            } else {
                throw NSError(domain: "AudioUpload", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "Upload failed with status code: \(httpResponse.statusCode)"])
            }
        } catch {
            print("Upload error: \(error)")
            throw error
        }
    }
    
    func startRecording() {
        let recordingSession = AVAudioSession.sharedInstance()
        
        do {
            try recordingSession.setCategory(.playAndRecord, mode: .default)
            try recordingSession.setActive(true)
            
            let documentPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let audioFilename = documentPath.appendingPathComponent("\(Date().toString()).m4a")
            
            let settings = [
                AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
                AVSampleRateKey: 44100,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
            ]
            
            audioRecorder = try AVAudioRecorder(url: audioFilename, settings: settings)
            audioRecorder?.isMeteringEnabled = true
            audioRecorder?.record()
            isRecording = true
            
            startMonitoring()
        } catch {
            print("Could not start recording: \(error)")
        }
    }
    
    func stopRecording() {
        audioRecorder?.stop()
        isRecording = false
        stopMonitoring()
        fetchRecordings()
    }
    
    func startPlayback(audio: URL) {
        do {
            // If currently playing something, stop it
            if currentlyPlayingURL != nil {
                stopPlayback()
            }
            
            audioPlayer = try AVAudioPlayer(contentsOf: audio)
            audioPlayer?.isMeteringEnabled = true
            audioPlayer?.delegate = self
            
            // Set duration before playing
            duration = audioPlayer?.duration ?? 0
            currentTime = 0
            
            audioPlayer?.play()
            currentlyPlayingURL = audio
            isPlaying = true
            startMonitoring()
            startPlaybackTimer()
        } catch {
            print("Playback failed: \(error)")
        }
    }
    
    func pausePlayback() {
        audioPlayer?.pause()
        isPlaying = false
        stopPlaybackTimer()
    }
    
    func resumePlayback() {
        audioPlayer?.play()
        isPlaying = true
        startPlaybackTimer()
    }
    
    func stopPlayback() {
        audioPlayer?.stop()
        audioPlayer?.delegate = nil
        currentlyPlayingURL = nil
        isPlaying = false
        currentTime = 0
        duration = 0
        stopMonitoring()
        stopPlaybackTimer()
        currentPower = 0.0
    }
    
    func seek(to time: TimeInterval) {
        audioPlayer?.currentTime = time
        currentTime = time
    }
    
    private func startPlaybackTimer() {
        playbackTimer = Timer.scheduledTimer(withTimeInterval: 0.01, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            self.currentTime = self.audioPlayer?.currentTime ?? 0
        }
    }
    
    private func stopPlaybackTimer() {
        playbackTimer?.invalidate()
        playbackTimer = nil
    }
    
    private func startMonitoring() {
        timer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            self?.updatePower()
        }
    }
    
    private func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }
    
    private func updatePower() {
        if isRecording {
            audioRecorder?.updateMeters()
            let power = audioRecorder?.averagePower(forChannel: 0) ?? -160
            updateCurrentPower(power)
        } else if audioPlayer?.isPlaying == true {
            audioPlayer?.updateMeters()
            let power = audioPlayer?.averagePower(forChannel: 0) ?? -160
            updateCurrentPower(power)
        }
    }
    
    private func updateCurrentPower(_ power: Float) {
        let minDb: Float = -60
        
        if power < minDb {
            DispatchQueue.main.async {
                self.currentPower = 0.0
            }
            return
        }
        
        let normalizedValue = Double(pow(10, (power - minDb) / 20))
        DispatchQueue.main.async {
            self.currentPower = min(max(normalizedValue, 0), 1)
        }
    }
    
    private func fetchRecordings() {
        let documentPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        do {
            let urls = try FileManager.default.contentsOfDirectory(at: documentPath, includingPropertiesForKeys: nil)
            recordings = urls.filter { $0.pathExtension == "m4a" || $0.pathExtension == "mp3" || $0.pathExtension == "wav"  }
                .sorted(by: { $0.lastPathComponent > $1.lastPathComponent })
            print("Fetched recordings: \(recordings)")
        } catch {
            print("Could not fetch recordings: \(error)")
        }
    }
}

extension AudioRecorderViewModel: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        DispatchQueue.main.async {
            self.stopPlayback()
        }
    }
}
