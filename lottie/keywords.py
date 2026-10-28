"""Stage-1 caption/tag keyword lists copied from ../I2V/scripts/filter_common.py (recall-biased text prefilter)."""

PERSON_WORDS = [
    "person", "people", "human", "humans", "man", "men", "woman", "women", "boy", "boys",
    "girl", "girls", "child", "children", "kid", "kids", "baby", "babies", "infant", "toddler",
    "teen", "teenager", "teenagers", "adult", "elderly", "elder", "senior", "grandma", "grandpa",
    "grandmother", "grandfather", "mother", "father", "mom", "mum", "dad", "daddy", "mommy",
    "sister", "brother", "sibling", "family", "lady", "ladies", "gentleman", "guy", "guys",
    "worker", "employee", "doctor", "nurse", "teacher", "student", "chef", "cook", "farmer",
    "athlete", "dancer", "singer", "musician", "artist", "painter", "writer", "scientist",
    "engineer", "pilot", "driver", "police", "policeman", "policewoman", "firefighter", "soldier",
    "superhero", "hero", "heroine", "wizard", "witch", "king", "queen", "prince", "princess",
    "knight", "pirate", "ninja", "samurai", "astronaut", "businessman", "businesswoman", "waiter",
    "waitress", "barber", "builder", "mechanic", "surgeon", "patient", "customer", "shopper",
    "tourist", "traveler", "runner", "swimmer", "boxer", "wrestler", "gymnast", "cyclist",
    "skater", "skier", "climber", "hiker", "gamer", "graduate", "bride", "groom", "couple",
    "friend", "friends", "colleague", "boss", "manager", "clown", "magician",
]

ANIMAL_WORDS = [
    "animal", "animals", "pet", "pets", "dog", "dogs", "puppy", "puppies", "cat", "cats",
    "kitten", "kittens", "horse", "pony", "cow", "bull", "ox", "pig", "piglet", "sheep", "lamb",
    "goat", "rabbit", "bunny", "mouse", "mice", "rat", "hamster", "guinea pig", "squirrel", "fox",
    "wolf", "bear", "panda", "koala", "lion", "tiger", "leopard", "cheetah", "elephant", "giraffe",
    "zebra", "monkey", "ape", "gorilla", "chimpanzee", "deer", "moose", "reindeer", "camel",
    "kangaroo", "hedgehog", "otter", "raccoon", "skunk", "bat", "bird", "birds", "chicken",
    "rooster", "hen", "duck", "goose", "swan", "owl", "eagle", "hawk", "falcon", "parrot",
    "penguin", "flamingo", "peacock", "pigeon", "crow", "robin", "sparrow", "toucan", "fish",
    "shark", "whale", "dolphin", "octopus", "squid", "crab", "lobster", "shrimp", "turtle",
    "tortoise", "frog", "toad", "snake", "lizard", "gecko", "chameleon", "dinosaur", "dragon",
    "bee", "bees", "wasp", "butterfly", "moth", "ant", "ants", "spider", "ladybug", "beetle",
    "snail", "worm", "insect", "insects", "bug", "bugs", "caterpillar", "dragonfly", "cricket",
    "grasshopper", "scorpion", "crocodile", "alligator", "hippo", "rhino", "donkey", "llama",
    "alpaca", "seal", "walrus", "jellyfish", "starfish", "seahorse",
]
