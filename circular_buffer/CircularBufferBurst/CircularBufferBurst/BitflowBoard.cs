using BufferAcquisition;
using System;

namespace CircularBufferBurst
{
    public class BitFlowBoard
    {
        private const OverwriteMethod DefaultOverwriteMethod = OverwriteMethod.Ignore;
        private const SetupOptions DefaultAcquisitionOptions = SetupOptions.Default;
        private readonly CircularAcquisition _circAcq;

        public BitFlowBoard()
        {
            _circAcq = new CircularAcquisition();
            OpenAcquisition();
            // Change this to your own Camera File name or the program will not work.
            // Does not need the full path, only the file name
            _circAcq.LoadCameraFile("illunis-RMV-4070.bfml");
            SetUpAcquisitionOptions();
        }

        public CircularAcquisition CircularAcquisition { get { return _circAcq; } }

        private uint NumImageBuffers { get; set; }

        private void OpenAcquisition()
        {
            // Attempt to open exclusively to verify no other app in the system is using it
            try
            {
                _circAcq.Open(0, OpenOptions.Exclusive | OpenOptions.NoOpenErrorMess);
            }
            catch (ApplicationException)
            {
                throw new Exception("Could not open the frame grabber.  Check for another application using it");
            }
            catch (ArgumentException)
            {
                throw new Exception("Could not find the frame grabber");
            }
        }

        private void SetUpAcquisitionOptions()
        {
            _circAcq.Cleanup();

            // If we catch up to the tail of the buffer that position in the buffer will be overwritten.
            // It will continue to overwrite frames in the buffer until acquisition is stopped or whoever is writing
            // data out catches up to the acquisition
            _circAcq.SetOverwriteMethod(DefaultOverwriteMethod);

            _circAcq.SetSetupOptions(DefaultAcquisitionOptions);
            //Set to the appropriate frame width and height
            _circAcq.SetAcqFrameSize((uint)1024, (uint)1024);
            _circAcq.SetAcqROI(0, 0, (uint)1024, (uint)1024);
            _circAcq.Cleanup();
        }
    }
}