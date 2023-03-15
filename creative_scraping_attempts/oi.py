import asyncio
import aiohttp
import pandas as pd


async def fetch_data(subreddit, params, last_utc):
    base_url = "https://api.pushshift.io/reddit/search/submission"
    p_string = "&".join([f"{k}={v}" for k, v in params.items()])

    if last_utc:
        # update the p-string to get the next batch
        params["before"] = last_utc
        p_string = "&".join([f"{k}={v}" for k, v in params.items()])

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{base_url}?subreddit={subreddit}&size=500&{p_string}"
        ) as response:
            data = await response.json()
            return pd.DataFrame(data["data"])


async def get_posts(subreddit: str, n: int, params: dict) -> pd.DataFrame:
    """Download posts from a subreddit using the Pushshift API

    Args:
        subreddit (str): subreddit to scrape
        n (int): number requests to make (500 posts per request)
        params (dict): dictionary of parameters to pass to the API excluding subreddit and size
    """
    dfs = []
    last_utc = None

    tasks = []
    for i in range(n):
        task = asyncio.create_task(fetch_data(subreddit, params, last_utc))
        tasks.append(task)

        df = await task
        if df.empty:
            break

        dfs.append(df)

    return pd.concat(dfs).drop_duplicates(subset="title").reset_index(drop=True)


async def run():
    subreddit = "conspiracy"
    n = 100
    params = {"sort_type": "score", "metadata": False, "after": "1y"}
    df = await get_posts(subreddit, n, params)
    print(df.shape)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
