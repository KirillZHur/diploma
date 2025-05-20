from data_io import plot_profiles

if __name__ == "__main__":
    for x_target in (1.0, 2.0):
        print(f"--- Profiles at x = {x_target} m ---")
        plot_profiles(x_target, Ny_list=[10])
