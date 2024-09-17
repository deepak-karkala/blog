# Content Popularity for CDN

- How is content popularity used to optimize our CDN?
	- Minimizing network distance
		- Given the finite amount of disk space available per server and the large size of the entire Netflix catalog, we cannot fit all content in every cluster of co-located servers.
		- Many clusters that are proximally located to end-users (ISP clusters) do not have enough disk capacity to fit the entire Netflix catalog. Therefore, we cache only the most popular content on these clusters.

	- Organizing content into server tiers
		- At locations that deliver very large amounts of traffic, we use a tiered infrastructure — high throughput servers (up to 100Gbps) are used to serve very popular content and high capacity storage servers (200TB+) are used to serve the tail of the catalog.
		- We need to rank content based on popularity to properly organize it within these tiers.

	- Influencing content replication within a cluster
		- Within a cluster, we replicate titles over N servers, where N is roughly proportional to the popularity of that content.

- Why do we store multiple copies of our files?
	- An extremely popular file, if deployed only on a single server, can overwhelm the resources of that server — while other servers may remain underutilized. This effect is not as pronounced in our deployment environment due to two crucial optimizations:
		- Because we route traffic based on network proximity, the regional demand for even the most popular content gets shared and diffused across our network.
		- Popular files are locked into memory rather than fetched constantly from disk. This latter memory optimization eliminates the possibility of disk I/O being the cause of a server capacity bottleneck.

	- Maximizing traffic by minimizing inter-server traffic variance
		- Consistent Hashing is used to allocate content to multiple servers within a cluster. While consistent hashing on its own typically results in a reasonably well-balanced cluster, the absolute traffic variance can be high if every file is served from a single server in a given location.
		- high popularity content (large rocks) can be broken down into less popular content (pebbles) simply by deploying multiple copies of this content.

	- Resilience to server failures and unexpected spikes in popularity
		- In the event that a server has failed, all of the traffic bound to that server needs to be delivered from other servers in the same cluster — or, from other more distant locations on the network. Staying within the same cluster, therefore minimizing network distance, is much preferable

- How is our content organized?
	- Every title is encoded in multiple formats, or encoding profiles. For example, some profiles may be used by iOS devices and others for a certain class of Smart TVs. There are video profiles, audio profiles, and profiles that contain subtitles.
	- Which bitrate you stream at depends on the quality of your network connection, the encoding profiles your device supports, the title itself, and the Netflix plan that you are subscribed to.
	- So for each quadruple of (title, encoding profile, bitrate, language), we need to cache one or more files. As an example, for streaming one episode of The Crown we store around 1,200 files!

- How do we evaluate content ranking effectiveness?
	- <b>Caching efficiency</b>
		- For a cluster that is set up to service a certain segment of our traffic, caching efficiency is the ratio of bytes served by that cluster versus overall bytes served for this segment of traffic.
	- Maximizing caching efficiency at the closest possible locations translates to lesser network hops. Lesser network hops directly improves user streaming quality, and also reduces the cost of transporting network content for both ISP networks and Netflix.

- How do we predict popularity?
	- we predict future viewing patterns by looking at historical viewing patterns.
	- we smooth data collected over multiple days of history to make the best prediction for the next day.

- What granularity do we use to predict popularity?
	- Title level: Using this ranking for content positioning causes all files associated with a title to be ranked in a single group.
	- File level: Every file is ranked on its own popularity. Using this method, files from the same title are in different sections of the rank. However, this mechanism improves caching efficiency significantly.
	- Orthogonally to the above 2 levels of aggregation, we compute content popularity on a regional level. This is with the intuitive presumption that members from the same country have a similar preference in content.