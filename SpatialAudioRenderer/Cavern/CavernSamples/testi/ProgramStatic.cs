using System;
using System.Collections.Generic;
using System.IO;
using Cavern;
using Cavern.Format.Environment;
using Cavern.Format;
using System.Numerics;
using Cavern.Channels;
using Cavern.Format.Consts;
using Microsoft.VisualBasic.FileIO;
using System.Runtime.InteropServices;

namespace testi{

    static class ProgramStatic{

        static void MainL(){



            Clip clip = AudioReader.ReadClip("onlyRg.wav");
        

            int sprate = clip.SampleRate;

            // Create a rendering environment/Listener
            Listener listener = new Listener(){
                SampleRate = sprate,
                UpdateRate = 250,
            };


            Source source = new() {
                Clip = clip,
                DistanceSimulation = true,
            };
            listener.AttachSource(source);
            
            long length = (long) clip.Length*sprate;
            // Console.WriteLine(length);


            // BroadcastWaveFormatWriter writer = new BroadcastWaveFormatWriter("audio_spatial.wav", listener, length, BitDepth.Int16);
            (ReferenceChannel, Source)[] staticObjects = new (ReferenceChannel, Source)[0];
            DolbyAtmosBWFWriter writer = new DolbyAtmosBWFWriter("confcall/only_RgConfcall_spatial_depth.wav", listener, length, BitDepth.Int16, staticObjects);

            // To contain object movement in the file, it has to be written frame-by-frame
            long progress = 0;
            while (progress < length) {

                source.Position = new Vector3((float)(-0.25), (float)(0.3), (float)(-0.4)) * Listener.EnvironmentSize;

                // WriteNextFrame will update the Listener, you don't have to
                writer.WriteNextFrame();

                progress += listener.UpdateRate;
            }
            writer.Dispose(); // Disposal will append the object movement to the WAV file
        }
    }
}