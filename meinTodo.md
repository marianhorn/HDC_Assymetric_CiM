#TODO





    - neuer branch:
        - ga training data von hdc training data trennen WICHITG!!
        - Cross training verschiedener datasets testen


Crossover summary: 
Implement generation-dependent chunk sizes for grouped event-list crossover. The current implementation uses a fixed block size; replace this with a function that, for a given generation, iteratively returns chunk sizes until the full event list of length K is partitioned.

Definitions:

K = GA_MAX_FLIPS_CIM = total number of events in the event list

G = GA_DEFAULT_GENERATIONS = total number of generations

g = current generation, 0 <= g < G

Design:

The average chunk size should grow over generations:

start at 1% of K

end at 15% of K

Use:

mu_min = max(2, round(0.01 * K))

mu_max = max(mu_min + 1, round(0.15 * K))

Generation progress:

t = g / (G - 1) for G > 1, else t = 1

Geometric schedule for the average chunk size:

mu(g) = mu_min * pow((double)mu_max / mu_min, pow(t, alpha))

Parameter:

alpha controls how fast chunk size grows over generations

alpha = 1.0 → standard geometric growth

alpha > 1.0 → slower growth early, faster later

alpha < 1.0 → faster growth early

use a configurable constant, default alpha = 1.5

Sampling:

Sample actual chunk sizes from a uniform integer distribution around mu(g)

Use relative width r = 0.4

c_min = max(1, floor(mu * (1 - r)))

c_max = max(c_min, floor(mu * (1 + r)))

sample with rand() % (c_max - c_min + 1)

Partitioning:

Repeatedly sample chunk sizes until the whole list is consumed

Let remaining be the number of events not yet assigned to a block

If sampled chunk size > remaining, clamp it to remaining

Return blocks iteratively until remaining == 0

Implementation requirements:

efficient C implementation

no extra libraries beyond standard C

ideally precompute mu(g) for all generations once during GA initialization to avoid repeated pow() calls during crossover

Goal:

early generations: small chunks → more mixing/exploration

late generations: larger chunks → more structure preservation/exploitation







Offset Summary: 

Add a random offset before chunking to avoid first-chunk bias.

Implement generation-dependent chunking for grouped event-list crossover as follows:

Use:

K = GA_MAX_FLIPS_CIM = number of events in the event list

G = GA_DEFAULT_GENERATIONS = total number of generations

g = current generation, 0 <= g < G

First, rotate the event list by a random offset:

sample offset = rand() % K

build the shifted list as:

event_list[offset ... K-1]

followed by event_list[0 ... offset-1]

This means the chunking starts at a random position in the sorted event list, and the skipped prefix is appended at the end. This avoids systematic bias on the first chunk.

Then compute the average chunk size for generation g:

mu_min = max(2, round(0.01 * K))

mu_max = max(mu_min + 1, round(0.15 * K))

Normalize generation progress:

if G > 1: t = (double)g / (double)(G - 1)

else: t = 1.0

Use geometric scheduling:

mu = mu_min * pow((double)mu_max / (double)mu_min, pow(t, alpha))

Parameter:

alpha controls how fast the average chunk size grows

alpha = 1.0 → standard geometric growth

alpha > 1.0 → slower growth early, faster later

alpha < 1.0 → faster growth early

default: alpha = 1.5

Sample actual chunk sizes uniformly around mu:

use r = 0.4

c_min = max(1, floor(mu * (1 - r)))

c_max = max(c_min, floor(mu * (1 + r)))

sample:

chunk_size = c_min + rand() % (c_max - c_min + 1)

Then partition the shifted event list iteratively:

initialize remaining = K

while remaining > 0:

sample chunk_size

if chunk_size > remaining, set chunk_size = remaining

return/output this chunk size

subtract it from remaining

So the full process is:

rotate the event list by a random offset

compute generation-dependent average chunk size

sample chunk sizes uniformly around that average

iteratively partition the shifted list until all K events are assigned to chunks

Expected behavior:

early generations: small chunks, more mixing

late generations: larger chunks, more structure preservation

random offset prevents systematic first-chunk bias

Implementation notes:

keep it efficient in C

use only standard functions like rand() and pow()

optionally precompute the scheduled means mu[g] once at initialization to avoid repeated pow() calls during runtime







Mutation summary: 