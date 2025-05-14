        # test_dns_aiohttp.py
import asyncio
import aiohttp

async def main():
    try:
        async with aiohttp.ClientSession() as session:
                    # Test Google DNS resolution first, then a target site
            async with session.get('https://dns.google.com') as response: # Test a known simple HTTPS endpoint
                print(f"Status for dns.google.com: {response.status}")
            async with session.get('https://apple.com') as response:
                print(f"Status for apple.com: {response.status}")
            async with session.get('https://generativelanguage.googleapis.com') as response:
                print(f"Status for generativelanguage.googleapis.com: {response.status}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())