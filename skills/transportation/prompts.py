TRANSPORTATION_INTENT_PROMPT = """Extract the origin, destination, and travel mode from the user's navigation query.

Defaults (apply when not explicitly stated):
- origin: London
- mode: public-transport

Travel modes:
- public-transport: buses, tubes, trains — use unless another mode is specified
- driving-car: driving, by car
- foot-walking: walking, on foot
- cycling-regular: cycling, by bike

Examples:
- "how do I get to Chelsea" → origin: London, destination: Chelsea, mode: public-transport
- "how do I get from Putney to Chelsea" → origin: Putney, destination: Chelsea, mode: public-transport
- "drive from Manchester to Leeds" → origin: Manchester, destination: Leeds, mode: driving-car
- "walk from Victoria station to Buckingham Palace" → origin: Victoria station, destination: Buckingham Palace, mode: foot-walking
- "cycle from Shoreditch to London Bridge" → origin: Shoreditch, destination: London Bridge, mode: cycling-regular
- "directions to Heathrow" → origin: London, destination: Heathrow airport, mode: public-transport
"""
