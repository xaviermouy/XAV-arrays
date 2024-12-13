import ecosound.core.tools
from ecosound.core.metadata import DeploymentInfo
from ecosound.core.audiotools import Sound
from ecosound.core.measurement import Measurement
import ecosound
import os


def process_folder(in_dir):
    files = ecosound.core.tools.list_files(
            in_dir,
            ".nc",
            recursive=False,
            case_sensitive=True,
        )

    first = True
    for idx, in_file in enumerate(files):
            print(in_file)
            meas = Measurement()
            meas.from_netcdf(in_file)
            if first == True:
                All = meas
            else:
                try:
                    All = All + meas
                except:
                    print('Could not load localization')
            first = False

    outfile = os.path.join(in_dir,All.data.iloc[0]['audio_file_name']+All.data.iloc[0]['audio_file_extension'])
    outdir = os.path.split(outfile)[0]

    All.to_netcdf(outfile+'.nc')
    All.to_raven(outdir)

# def test_files(in_dir):
#     garbage_dir = r'C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\Taylor-Islet_LA_dep2\results\results\garbage'
#     files = ecosound.core.tools.list_files(
#             in_dir,
#             ".nc",
#             recursive=False,
#             case_sensitive=True,
#         )

    first = True
    for idx, in_file in enumerate(files):
            #print(in_file)
            file = os.path.split(in_file)[1]
            print(file)
            meas = Measurement()
            meas.from_netcdf(in_file)
            try:
                meas.to_netcdf(os.path.join(garbage_dir,file))
                meas.to_raven(os.path.join(garbage_dir))
            except:
                print(file)
                continue


main_dir = r'C:\Users\xavier.mouy\Documents\Projects\2023_LizardIsland_AIMS\analysis\Localization\results\tmp'

for single_dir in os.listdir(main_dir):
    in_dir = os.path.join(main_dir,single_dir)
    print(in_dir)
    try:
        process_folder(in_dir)
        #test_files(in_dir)
    except:
        print('failled')