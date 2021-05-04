import json

import requests
from pyyoutube import Api


def get_data(YOUTUBE_API_KEY, videoId, maxResults, nextPageToken):
    """
    Получение информации со страницы с видео по video id
    """
    YOUTUBE_URI = 'https://www.googleapis.com/youtube/v3/commentThreads?key={KEY}&textFormat=plainText&' + \
                  'part=snippet&videoId={videoId}&maxResults={maxResults}&pageToken={nextPageToken}'
    format_youtube_uri = YOUTUBE_URI.format(KEY=YOUTUBE_API_KEY,
                                            videoId=videoId,
                                            maxResults=maxResults,
                                            nextPageToken=nextPageToken)
    content = requests.get(format_youtube_uri).text
    data = json.loads(content)
    return data


def get_text_of_comment(data):
    """
    Получение комментариев из полученных данных под одним видео
    """
    comms = set()
    for item in data['items']:
        comm = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comms.add(comm)
    return comms


def get_all_comments(YOUTUBE_API_KEY, query, count_video=10, limit=30, maxResults=10, nextPageToken=''):
    """
    Выгрузка maxResults комментариев
    """
    api = Api(api_key=YOUTUBE_API_KEY)
    video_by_keywords = api.search_by_keywords(q=query,
                                               search_type=["video"],
                                               count=count_video,
                                               limit=limit)
    videoId = [x.id.videoId for x in video_by_keywords.items]

    comments_all = []
    for id_video in videoId:
        try:
            data = get_data(YOUTUBE_API_KEY,
                            id_video,
                            maxResults=maxResults,
                            nextPageToken=nextPageToken)
            comment = list(get_text_of_comment(data))
            comments_all.append(comment)
        except:
            continue
    comments = sum(comments_all, [])
    return comments
