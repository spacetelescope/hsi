using Microsoft.Practices.Prism.Commands;

namespace CircularBufferBurst
{
    internal class CircularBufferBurstViewModel
    {
        private BitFlowBoard _bitflowBoard;
        private BitflowBurstAcquisiton _bitflowBurstAcq;
        private DelegateCommand _StartAcquisitionCommand;
        private DelegateCommand _StopAcquisitionCommand;

        public CircularBufferBurstViewModel()
        {
            _bitflowBoard = new BitFlowBoard();
            _bitflowBurstAcq = new BitflowBurstAcquisiton(_bitflowBoard);
        }

        public DelegateCommand StartAcquisiontionCommand
        {
            get
            {
                if (_StartAcquisitionCommand == null)
                {
                    _StartAcquisitionCommand = new DelegateCommand(StartAcquisiton, () => true);
                }
                return _StartAcquisitionCommand;
            }
        }

        public DelegateCommand StopAcquisiontionCommand
        {
            get
            {
                if (_StopAcquisitionCommand == null)
                {
                    _StopAcquisitionCommand = new DelegateCommand(StopAcquisiton, () => true);
                }
                return _StopAcquisitionCommand;
            }
        }

        private void StopAcquisiton()
        {
            _bitflowBurstAcq.StopAcquiring();
        }

        private void StartAcquisiton()
        {
            _bitflowBurstAcq.Acquire();
        }
    }
}