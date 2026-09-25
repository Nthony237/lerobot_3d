import pytest

from lerobot_3d.recording.episode_controller import EpisodeController, EpisodeState

pytestmark = pytest.mark.pure


class _FakeRecorder:
    def __init__(self):
        self.num_saved_episodes = 0
        self.frames = 0
        self.calls = []

    def start_episode(self):
        self.calls.append("start")
        self.frames = 0

    def add_frame(self, datapoints, snapshot, action):
        self.frames += 1

    def stop_episode(self):
        self.calls.append("stop")
        return self.frames

    def save_episode(self):
        self.calls.append("save")
        self.num_saved_episodes += 1
        self.frames = 0

    def discard_episode(self):
        self.calls.append("discard")
        self.frames = 0

    def finalize(self):
        self.calls.append("finalize")


def _controller():
    recorder = _FakeRecorder()
    return EpisodeController(recorder, fps=10, log=lambda _msg: None), recorder


def _step(controller, n=1):
    for _ in range(n):
        controller.on_step([], None, {})


def test_enter_space_y_saves_episode():
    controller, recorder = _controller()

    controller.handle_key("\n")
    _step(controller, 3)
    controller.handle_key(" ")
    assert controller.state is EpisodeState.CONFIRMING
    controller.handle_key("y")

    assert recorder.calls == ["start", "stop", "save"]
    assert recorder.num_saved_episodes == 1
    assert controller.state is EpisodeState.IDLE


def test_enter_space_n_discards_episode():
    controller, recorder = _controller()

    controller.handle_key("\n")
    _step(controller, 3)
    controller.handle_key(" ")
    controller.handle_key("N")

    assert recorder.calls == ["start", "stop", "discard"]
    assert recorder.num_saved_episodes == 0


def test_frames_only_recorded_while_recording():
    controller, recorder = _controller()

    _step(controller, 2)  # idle
    controller.handle_key("\r")
    _step(controller, 3)
    controller.handle_key(" ")
    _step(controller, 2)  # confirming

    assert recorder.frames == 3


def test_empty_episode_is_discarded_without_prompt():
    controller, recorder = _controller()

    controller.handle_key("\n")
    controller.handle_key(" ")

    assert recorder.calls == ["start", "stop", "discard"]
    assert controller.state is EpisodeState.IDLE


def test_irrelevant_keys_ignored():
    controller, recorder = _controller()

    controller.handle_key(" ")  # space while idle
    controller.handle_key("y")  # y while idle
    controller.handle_key("\n")
    controller.handle_key("\n")  # enter while recording
    _step(controller)
    controller.handle_key(" ")
    controller.handle_key("x")  # not y/n while confirming

    assert recorder.calls == ["start", "stop"]
    assert controller.state is EpisodeState.CONFIRMING


@pytest.mark.parametrize("keys", [["\n"], ["\n", " "]])
def test_close_mid_episode_discards_then_finalizes(keys):
    controller, recorder = _controller()
    controller.handle_key(keys[0])
    _step(controller)
    for key in keys[1:]:
        controller.handle_key(key)

    controller.close()

    assert recorder.calls[-2:] == ["discard", "finalize"]


def test_close_when_idle_only_finalizes():
    controller, recorder = _controller()

    controller.close()

    assert recorder.calls == ["finalize"]


def test_elapsed_counts_recorded_frames_and_resets_per_episode():
    controller, _recorder = _controller()

    controller.handle_key("\n")
    _step(controller, 25)
    assert controller.elapsed_s == pytest.approx(2.5)
    assert "● REC episode 0  00:02.5" == controller.status_text()

    controller.handle_key(" ")
    _step(controller, 5)  # not recording: timer frozen
    assert controller.elapsed_s == pytest.approx(2.5)
    assert "00:02.5" in controller.status_text()
    controller.handle_key("y")

    controller.handle_key("\n")
    assert controller.elapsed_s == 0.0
    assert controller.status_text() == "● REC episode 1  00:00.0"


def test_status_text_idle():
    controller, _recorder = _controller()

    assert controller.status_text().startswith("Idle -- 0 episodes saved")


@pytest.mark.parametrize(
    "seconds, expected", [(0, "00:00.0"), (9.94, "00:09.9"), (61.25, "01:01.2"), (3725, "62:05.0")]
)
def test_format_duration(seconds, expected):
    from lerobot_3d.recording.episode_controller import format_duration

    assert format_duration(seconds) == expected
