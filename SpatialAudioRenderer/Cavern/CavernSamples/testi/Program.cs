using System;
using System.Collections.Generic;
using System.IO;
using Cavern;
using Cavern.Format.Environment;
using Cavern.Format;
using System.Numerics;
using Cavern.Channels;
using Cavern.Format.Consts;

namespace testi{

    static class Program{

        static void Main(){

            Clip clip = AudioReader.ReadClip("piano2.wav");

            // Listener.HeadphoneVirtualizer = true;

            Console.WriteLine(clip.SampleRate);

            // Create a rendering environment with a single audio source
            // that's playing a second of sine wave
            Listener listener = new Listener(){
                SampleRate = clip.SampleRate,
                UpdateRate = 256
            };


            // float[] sine = new float[listener.SampleRate];
            // float mul = 2 * MathF.PI * 1000 / listener.SampleRate;
            // for (int i = 0; i < sine.Length; i++) {
            //     sine[i] = MathF.Sin(mul * i);
            // }
            Source source = new() {
                Clip = clip,
                // Position = new Vector3(10, 0, 0) * Listener.EnvironmentSize
            };
            listener.AttachSource(source);

            // Create a 10 second ADM BWF file
            long length = (long) clip.Length * clip.SampleRate;
            // BroadcastWaveFormatWriter writer = new BroadcastWaveFormatWriter("test_fixed.wav", listener, length, BitDepth.Int16);

            // BroadcastWaveFormatWriter writer = new BroadcastWaveFormatWriter("test.xml", listener, length, BitDepth.Int16);

            (ReferenceChannel, Source)[] staticObjects = new (ReferenceChannel, Source)[0];
            DolbyAtmosBWFWriter writer = new DolbyAtmosBWFWriter("test_dolby.wav", listener, length, BitDepth.Int16, staticObjects);

            // To contain object movement in the file, it has to be written frame-by-frame
            long progress = 0;
            while (progress < length) {
                // Example circular object motion
                float t = 2 * MathF.PI * progress / listener.SampleRate;
                Vector3 referencePos = new Vector3(MathF.Sin(t), MathF.Sin(t * .1f), MathF.Cos(t));
                // All positions will be written relative to the environment size
                source.Position = referencePos * Listener.EnvironmentSize;
                // WriteNextFrame will update the Listener, you don't have to
                writer.WriteNextFrame();

                progress += listener.UpdateRate;
            }
            writer.Dispose(); // Disposal will append the object movement to the WAV file
        }
    }
}