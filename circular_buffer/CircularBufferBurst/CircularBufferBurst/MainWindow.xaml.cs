using System.Windows;

namespace CircularBufferBurst
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            var vm = new CircularBufferBurstViewModel();
            DataContext = vm;
        }
    }
}