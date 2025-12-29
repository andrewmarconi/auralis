# SoundFont Files

## FluidR3_GM.sf2

This directory should contain the FluidR3_GM General MIDI SoundFont file for FluidSynth rendering.

### File Details
- **Filename**: `FluidR3_GM.sf2`
- **Size**: ~141 MB (too large for standard git)
- **Location**: `soundfonts/FluidR3_GM.sf2`
- **Purpose**: General MIDI instrument sounds for music generation

### Download Instructions

The SoundFont file is excluded from git due to its large size (exceeds GitHub's 100MB limit).

**Download the FluidR3_GM SoundFont:**

1. Visit: https://github.com/FluidSynth/fluidsynth/wiki/SoundFont
2. Download FluidR3_GM.sf2 (approximately 141 MB)
3. Place it in `soundfonts/soundfonts/FluidR3_GM.sf2`

**Or use wget:**
```bash
cd soundfonts/soundfonts
wget https://keymusician.com/FluidR3_GM/FluidR3_GM.zip
unzip FluidR3_GM.zip
```

### Alternative: Use Git LFS

For team development, consider using Git Large File Storage (LFS):

```bash
# Install Git LFS
brew install git-lfs  # macOS
# or: sudo apt install git-lfs  # Linux

# Initialize Git LFS in repository
git lfs install

# Track SF2 files
git lfs track "*.sf2"

# Add the file
git add soundfonts/soundfonts/FluidR3_GM.sf2
git commit -m "Add SoundFont via Git LFS"
git push
```

## File Structure

```
soundfonts/
├── README.md                      # This file
├── .env.example                   # SoundFont configuration template
└── soundfonts/
    └── FluidR3_GM.sf2            # Downloaded SoundFont (not in git)
```

## Troubleshooting

**Error: SoundFont file not found**
- Ensure FluidR3_GM.sf2 is in `soundfonts/soundfonts/` directory
- Check file permissions are readable (chmod 644)
- Verify file size is approximately 141 MB

**Error: File exceeds GitHub's file size limit**
- The .sf2 file is in .gitignore and should not be committed
- If already committed, remove with: `git rm --cached soundfonts/soundfonts/*.sf2`
- Consider using Git LFS for team repositories
