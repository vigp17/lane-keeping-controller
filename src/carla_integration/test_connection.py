import carla


def main() -> None:

    # Connect to the running CARLA server
    client = carla.Client("localhost", 2000)

    # Set connection timeout
    client.set_timeout(10.0)

    # Retrieve the current simulation world
    world = client.get_world()

    # Get map information
    carla_map = world.get_map()

    print("Successfully connected to CARLA!")
    print(f"Map: {carla_map.name}")

    # Get available vehicle blueprints
    blueprint_library = world.get_blueprint_library()

    vehicles = blueprint_library.filter("vehicle.*")

    print(f"Available vehicle blueprints: {len(vehicles)}")


if __name__ == "__main__":
    main()