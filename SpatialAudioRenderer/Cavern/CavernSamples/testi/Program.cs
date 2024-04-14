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

namespace testi{

    static class Program{

        static void Main(){


            // all audio clips need to have the same sample rate
            // all audio clips should be same length, non speaker part needs to be silent
            // all co-ordinates need to be scaled between -1 and 1
            // make sure the co-ordinates are samples at a high rate (min 1FPS)
            // read the config file, get path to audio file and co-ordinates of the source
            string filePath = "data.csv";
            int coordSampleRate = 24;

            List<string> audioPaths = new List<string>();
            List<List<float[]>> cValues = new List<List<float[]>>();

            using (TextFieldParser parser = new TextFieldParser(filePath))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");

                while (!parser.EndOfData)
                {
                    string[] fields = parser.ReadFields();

                    if (fields != null && fields.Length == 1 )
                    {
                        // Assuming the first row is the co-ord/frame sample rate
                        coordSampleRate = int.Parse(fields[0]);
                        continue;
                    }

                    if (fields != null)
                    {
                        // Assuming the first field is the audio file path
                        audioPaths.Add(fields[0]);

                        // Initialize lists to hold coord values for this row
                        List<float[]> cRowValues = new List<float[]>();

                        // Loop through the remaining fields
                        for (int i = 1; i < fields.Length; i += 3)
                        {
                            float[] cValuesTuple = new float[3];

                            // Parse coord values
                            if (i + 2 < fields.Length)
                            {
                                float.TryParse(fields[i], out cValuesTuple[0]);
                                float.TryParse(fields[i + 1], out cValuesTuple[1]);
                                float.TryParse(fields[i + 2], out cValuesTuple[2]);
                                cRowValues.Add(cValuesTuple);
                            }

                        }

                        // Add the coord values for this row to the main lists
                        cValues.Add(cRowValues);
                    }
                }
            }

            // Print the extracted data for verification
            Console.WriteLine("Audio Paths:");
            foreach (string audioPath in audioPaths)
            {
                Console.WriteLine(audioPath);
            }

            Console.WriteLine("\nC Values:");
            foreach (var cRow in cValues)
            {
                foreach (var cTuple in cRow)
                {
                    Console.WriteLine($"({cTuple[0]}, {cTuple[1]}, {cTuple[2]})");
                }
                Console.WriteLine("One wav completed\n");
            }

            // creating all the clips
            List<Clip> clips = new List<Clip>();
            int sprate = -1;
            foreach (string audioPath in audioPaths)
            {
                Clip clip = AudioReader.ReadClip(audioPath);
                
                // checking if all the clips have the same sample rate
                if (sprate == -1)
                {
                    sprate = clip.SampleRate;
                }
                else if (sprate != clip.SampleRate)
                {
                    Console.WriteLine("All clips must have the same sample rate");
                    return;
                }

                clips.Add(clip);
            }


            // headphone virtualizer
            // Listener.HeadphoneVirtualizer = true;
            
            Console.WriteLine(clips[0].SampleRate);

            // setting updateRate
            float updateTime = 1f / coordSampleRate;    
            updateRate = (int) updateTime * sprate;

            // Create a rendering environment/Listener
            Listener listener = new Listener(){
                SampleRate = sprate,
                UpdateRate = updateRate,
            };


            // adding sources
            List<Source> sources = new List<Source>();
            foreach (Clip clip in clips)
            {
                Source source = new() {
                    Clip = clip,
                };
                listener.AttachSource(source);
                sources.Add(source);
            }


            // Create a ADM BWF file
            // Get max length among all clips
            long maxLength = 0;
            foreach (Clip clip in clips)
            {
                if (clip.Length > maxLength)
                {
                    maxLength = clip.Length;
                }
            }
            long length = (long) maxLength * clip.SampleRate;



            // BroadcastWaveFormatWriter writer = new BroadcastWaveFormatWriter("test_fixed.wav", listener, length, BitDepth.Int16);
            (ReferenceChannel, Source)[] staticObjects = new (ReferenceChannel, Source)[0];
            DolbyAtmosBWFWriter writer = new DolbyAtmosBWFWriter("test.wav", listener, length, BitDepth.Int16, staticObjects);

            // To contain object movement in the file, it has to be written frame-by-frame
            long progress = 0;
            while (progress < length) {

                // check if all sources are playing and update positions
                for (int i = 0; i < sources.Count; i++)
                {
                    Source source = sources[i];
                    // if (!source.IsPlaying)
                    // {
                    //     source.listener.DetachSource(source); // may or may not work
                    //     continue;
                    // }
                    
                    // get the coord values for this frame
                    List<float[]> cRow = cValues[i];
                    if ((cRow.Count-1) <= ((int) (progress / updateRate)))
                    {
                        float[] cValues = cRow[(int) (progress / updateRate)]; // may or may not work

                        // All positions will be written relative to the environment size
                        source.Position = new Vector3(cValues[0], cValues[1], cValues[2]) * Listener.EnvironmentSize;
                    }
                    
                }


                // WriteNextFrame will update the Listener, you don't have to
                writer.WriteNextFrame();

                progress += listener.UpdateRate;
            }
            writer.Dispose(); // Disposal will append the object movement to the WAV file
        }
    }
}