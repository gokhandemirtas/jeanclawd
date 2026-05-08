from skills.base import Skill
from skills.spotify.skill import handle_spotify
from utils.auth.spotify import SpotifyAuthProvider

spotify_skill = Skill(
    id="SPOTIFY",
    description="Control Spotify playback — play, pause, skip, volume, now playing, queue",
    system_prompt=None,
    handle=handle_spotify,
    needs_wrapping=False,
    auth_provider=SpotifyAuthProvider(),
)
