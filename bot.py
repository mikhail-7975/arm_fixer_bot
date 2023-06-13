from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor

from config import TOKEN

import arm_fracture_segmentation as afs


bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def process_start_command(msg: types.Message):
    await msg.reply("Привет!\nПокажи мне свою руку и я найду на ней перелом!\nОтправь один или нескоько снимков")

@dp.message_handler(commands=['help'])
async def process_help_command(msg: types.Message):
    await msg.reply("Отправь мне снимок, на котором хочешь найти перелом!")

@dp.message_handler()
async def echo_message(msg: types.Message):
    await msg.reply("Отправь мне снимок, на котором хочешь найти перелом!")

@dp.message_handler(content_types=['photo'])
async def process_photo(msg: types.Message):
    photos = msg.photo
    await photos[-1].download("test.jpg")

    model = afs.Model()
    model.inference("test.jpg")

    with open('result.jpg', 'rb') as file:
        await bot.send_photo(msg.from_user.id, file)


if __name__ == '__main__':
    executor.start_polling(dp)
