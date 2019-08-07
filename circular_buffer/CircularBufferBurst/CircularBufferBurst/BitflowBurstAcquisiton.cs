using BufferAcquisition;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace CircularBufferBurst
{
    internal class BitflowBurstAcquisiton
    {
        private CircularAcquisition _circAcq;
        private int NumberOfFramesAcquired;
        private bool _keepLooping;
        private uint _numBuffers;
        private uint _currentBuffer;

        public BitflowBurstAcquisiton(BitFlowBoard bitflowBoard)
        {
            _circAcq = bitflowBoard.CircularAcquisition;
            SetupAcquisiton();
        }

        public void Acquire()
        {
            _circAcq.StartAcquisition(AcqControlOptions.Wait);
            _numBuffers = _circAcq.GetNumberOfBuffers();
            Trace.WriteLine("Number of buffers: " + _numBuffers);
            _currentBuffer = 0;
            NumberOfFramesAcquired = 0;
            _keepLooping = true;
            Trace.WriteLine("starting acq");
            Trace.WriteLine("Board is open: " + _circAcq.IsBoardOpen());
            Trace.WriteLine("Board is setup: " + _circAcq.IsBoardSetup());
            BufferInfo bufferInfo = new BufferInfo();
            new Thread(() =>
                {
                    Thread.CurrentThread.IsBackground = true;
                    try
                    {
                        while (_keepLooping)
                        {
                            // returns instantly if there is a frame in the buffer
                            // First parameter is the timeout, second is to get the buffer info back
                            var retval = _circAcq.WaitForFrameDone(1000, ref bufferInfo);
                            if (retval == WaitFrameDoneReturns.FrameAcquired)
                            {
                                Trace.WriteLine("Got a frame");
                                Trace.WriteLine("Current buffer position: " + _currentBuffer);
                                var bufferData = _circAcq.GetBufferData(_currentBuffer);
                                WriteOutFrame(bufferData, bufferInfo);
                                // Mark the buffer that we just wrote out as available
                                _circAcq.SetBufferStatus(_currentBuffer, BufferStatus.Available);
                                _currentBuffer++;
                                // Move back to the start of the buffers if we reach the end
                                _currentBuffer = _currentBuffer % _numBuffers;
                                NumberOfFramesAcquired++;
                                Trace.WriteLine("Number of frames Acquired: " + _circAcq.GetNumberOfFramesAcquired());
                                Trace.WriteLine("Number of frames Overwritten: " + _circAcq.GetNumberOfFramesOverwritten());
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        Trace.WriteLine("Num Frames Acquired before fail: " + _circAcq.GetNumberOfFramesAcquired()); ;
                        Trace.WriteLine("error : " + e.Message);
                        _circAcq.StopAcquisition(AcqControlOptions.Wait);
                    }
                }).Start();
        }

        internal void StopAcquiring()
        {
            _keepLooping = false;
            _circAcq.StopAcquisition(AcqControlOptions.Wait);
        }

        private static void WriteMemoryBlock(IEnumerable<byte> data, BinaryWriter writer)
        {
            foreach (var t in data)
            {
                writer.Write(t);
            }
        }

        private void WriteOutFrame(byte[] bufferData, BufferInfo bufferInfo)
        {
            using (var bw = new BinaryWriter(File.Open(CreatePathForSaving(), FileMode.Create)))
            {
                byte[] frameHeader = CreateFrameHeader(bufferInfo);
                WriteMemoryBlock(frameHeader, bw);

                var frameData = bufferData;
                WriteMemoryBlock(frameData, bw);

                bw.Close();
            }
            Trace.WriteLine("Wrote out frame: " + NumberOfFramesAcquired);
        }

        private byte[] CreateFrameHeader(BufferInfo bufferInfo)
        {
            byte[] header = new byte[24];

            byte[] frameNumber = BitConverter.GetBytes(NumberOfFramesAcquired);
            frameNumber.CopyTo(header, 0);

            byte[] hour = BitConverter.GetBytes(bufferInfo.m_TimeStamp.m_Hour);
            hour.CopyTo(header, 8);

            byte[] min = BitConverter.GetBytes(bufferInfo.m_TimeStamp.m_Min);
            min.CopyTo(header, 12);

            byte[] sec = BitConverter.GetBytes(bufferInfo.m_TimeStamp.m_Sec);
            sec.CopyTo(header, 16);

            byte[] uSec = BitConverter.GetBytes(bufferInfo.m_TimeStamp.m_uSec);
            uSec.CopyTo(header, 20);

            return header;
        }

        private string CreatePathForSaving()
        {
            //Change this to whatever directory you wish to save the frames to
            return "C:\\4D\\Data\\Burst\\frame" + NumberOfFramesAcquired;
        }

        private void SetupAcquisiton()
        {
            // you can also pass in your own list of intptr as a buffer instead of just using this setup method
            // ex: _circAcq.Setup(List<Intptr>(), 24); First parameter is a list of Intptr and the second parameter is the amount of buffers
            _circAcq.Setup(1024);
        }
    }
}