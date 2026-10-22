// The macOS Speech framework (SpeechAnalyzer, macOS 26 and later) as a helper process for audio/stt/apple.py.
//
// Protocol on stdin/stdout (see helper.py): one JSON line {"ready": true} at start; then messages of a
// one-byte command, a four-byte little-endian length and that many bytes of 16-bit mono PCM at 16 kHz.
// "T" transcribes a whole utterance and is answered with {"text": "..."}; "S", "F", "E" start, feed and
// end an utterance (frames are collected and transcribed at "E", which is answered). Errors come back
// as {"error": "..."}. Built on first use by apple.py with swiftc.

import AVFoundation
import Foundation
import Speech

func emit(_ object: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: object)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

func readExactly(_ count: Int) -> Data? {
    var data = Data()
    while data.count < count {
        let chunk = FileHandle.standardInput.readData(ofLength: count - data.count)
        if chunk.isEmpty { return nil }
        data.append(chunk)
    }
    return data
}

@main
struct Helper {
    static func main() async {
        do { try await run() } catch { emit(["error": "\(error)"]); exit(1) }
    }

    static func makeTranscriber() -> SpeechTranscriber {
        SpeechTranscriber(locale: Locale(identifier: "en_US"), transcriptionOptions: [], reportingOptions: [], attributeOptions: [])
    }

    static func run() async throws {
        let probe = makeTranscriber()
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [probe]) {
            try await request.downloadAndInstall()
        }
        guard let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [probe]) else {
            emit(["error": "no audio format for the transcriber"]); exit(1)
        }
        let pcmFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 16000, channels: 1, interleaved: true)!
        emit(["ready": true])
        var collected = Data()
        while true {
            guard let header = readExactly(5) else { return }
            let command = header[header.startIndex]
            let length = Int(header.subdata(in: header.startIndex + 1 ..< header.startIndex + 5).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }.littleEndian)
            guard let pcm = length > 0 ? readExactly(length) : Data() else { return }
            do {
                switch command {
                case UInt8(ascii: "T"):
                    emit(["text": try await transcribe(pcm, pcmFormat: pcmFormat, format: format)])
                case UInt8(ascii: "S"):
                    collected = Data()
                case UInt8(ascii: "F"):
                    collected.append(pcm)
                case UInt8(ascii: "E"):
                    emit(["text": try await transcribe(collected, pcmFormat: pcmFormat, format: format)])
                    collected = Data()
                default:
                    emit(["error": "unknown command \(command)"])
                }
            } catch {
                emit(["error": "\(error)"])
            }
        }
    }

    static func transcribe(_ pcm: Data, pcmFormat: AVAudioFormat, format: AVAudioFormat) async throws -> String {
        let frames = AVAudioFrameCount(pcm.count / 2)
        let inBuf = AVAudioPCMBuffer(pcmFormat: pcmFormat, frameCapacity: max(frames, 1))!
        inBuf.frameLength = frames
        pcm.withUnsafeBytes { raw in
            if let base = raw.baseAddress, frames > 0 {
                inBuf.int16ChannelData![0].update(from: base.assumingMemoryBound(to: Int16.self), count: Int(frames))
            }
        }
        let converter = AVAudioConverter(from: pcmFormat, to: format)!
        let capacity = AVAudioFrameCount(Double(frames) * format.sampleRate / pcmFormat.sampleRate) + 16
        let outBuf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity)!
        var fed = false
        var convError: NSError?
        converter.convert(to: outBuf, error: &convError) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return inBuf
        }
        if let e = convError { throw e }

        let transcriber = makeTranscriber()
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let (stream, continuation) = AsyncStream.makeStream(of: AnalyzerInput.self)
        let reader = Task { () throws -> String in
            var parts: [String] = []
            for try await result in transcriber.results where result.isFinal {
                parts.append(String(result.text.characters))
            }
            return parts.joined(separator: " ")
        }
        try await analyzer.start(inputSequence: stream)
        continuation.yield(AnalyzerInput(buffer: outBuf))
        continuation.finish()
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        return try await reader.value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
