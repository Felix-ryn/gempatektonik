using CommunityToolkit.Mvvm.Input;
using Tectonic.Models;

namespace Tectonic.PageModels
{
    public interface IProjectTaskPageModel
    {
        IAsyncRelayCommand<ProjectTask> NavigateToTaskCommand { get; }
        bool IsBusy { get; }
    }
}