{ pkgs }: {
  deps = [
    pkgs.python310Full
    pkgs.replitPackages.prybar-python310
    pkgs.replitPackages.stderred
    pkgs.ffmpeg
    pkgs.libsndfile
    pkgs.pkg-config
    pkgs.portaudio
    pkgs.alsaLib
  ];
  env = {
    PYTHON_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.libsndfile
      pkgs.ffmpeg
      pkgs.portaudio
      pkgs.alsaLib
    ];
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.libsndfile
      pkgs.ffmpeg
      pkgs.portaudio
      pkgs.alsaLib
    ];
  };
}
