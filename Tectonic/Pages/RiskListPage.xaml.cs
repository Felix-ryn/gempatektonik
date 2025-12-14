// Pages/RiskListPage.xaml.cs
using Tectonic.PageModels;

namespace Tectonic.Pages;

public partial class RiskListPage : ContentPage
{
    public RiskListPage(RiskListPageModel vm)
    {
        InitializeComponent();
        BindingContext = vm;
    }
}