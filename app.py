import os
import logging
import random
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from dotenv import load_dotenv

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load word lists at startup
WORDLIST_FILE = 'wordlist.txt'
OTHER_VALID_WORDS_FILE = 'other_valid_words.txt'

with open(WORDLIST_FILE, 'r') as f:
    WORDS = [w.strip().lower() for w in f if len(w.strip()) == 5]

# Load additional valid words for input validation (not for suggestions)
try:
    with open(OTHER_VALID_WORDS_FILE, 'r') as f:
        OTHER_VALID_WORDS = [w.strip().lower() for w in f if len(w.strip()) == 5]
except FileNotFoundError:
    OTHER_VALID_WORDS = []
    logger.warning(f"{OTHER_VALID_WORDS_FILE} not found. Only words from {WORDLIST_FILE} will be accepted.")

# Combined list for input validation
ALL_VALID_WORDS = set(WORDS + OTHER_VALID_WORDS)

# Feedback constants: 'g' = green/correct, 'y' = yellow/wrong position, 'b' = black/absent
GREEN = 'g'
YELLOW = 'y'
BLACK = 'b'

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /start and /new commands to begin a new game."""
    chat_id = update.effective_chat.id
    context.chat_data['candidates'] = WORDS.copy()
    context.chat_data['guesses'] = []
    context.chat_data['patterns'] = []
    await update.message.reply_text(
        'Welcome to Wordle Helper! I will suggest words for you.\nLet\'s begin.'
    )
    await suggest_next(update, context)

async def suggest_next(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Compute and display top 5 candidate words with likelihoods."""
    chat_id = update.effective_chat.id
    candidates = context.chat_data.get('candidates', WORDS)
    if not candidates:
        await context.bot.send_message(chat_id, 'No candidates remain! Try /new to start over.')
        return

    # Compute letter frequencies
    freq = {}
    for w in candidates:
        for ch in set(w):  # unique letters to reduce bias
            freq[ch] = freq.get(ch, 0) + 1

    # Score candidates
    scores = []
    for w in candidates:
        score = sum(freq[ch] for ch in set(w))
        scores.append((w, score))

    # Get top candidates for randomization
    top_candidates = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Randomize selection from top candidates
    if len(candidates) == len(WORDS):  # First turn - more randomization
        # Pick from top 20 candidates to add variety to starting words
        selection_pool = top_candidates[:min(20, len(top_candidates))]
        top5 = random.sample(selection_pool, min(5, len(selection_pool)))
    else:  # Later turns - still some randomization but favor better words
        # Pick from top 10 candidates for subsequent turns
        selection_pool = top_candidates[:min(10, len(top_candidates))]
        top5 = random.sample(selection_pool, min(5, len(selection_pool)))
    
    # Sort the selected 5 by score for display (highest first)
    top5 = sorted(top5, key=lambda x: x[1], reverse=True)
    
    # Calculate probabilities based only on top candidates for better display
    # This filters out very rare words from the probability calculation
    top_candidates_threshold = min(50, len(candidates))  # Use top 50 or all if fewer
    top_candidates_for_prob = sorted(scores, key=lambda x: x[1], reverse=True)[:top_candidates_threshold]
    top_total = sum(s for w, s in top_candidates_for_prob)
    
    # If we have very few candidates, use all of them for probability calculation
    if len(candidates) <= 10:
        prob_total = sum(s for w, s in scores)
    else:
        prob_total = top_total

    # Build suggestion buttons for inline keyboard
    suggestion_buttons = [
        InlineKeyboardButton(f"{w.upper()} ({s/prob_total:.1%})", callback_data=f"guess:{w}")
        for w, s in top5
    ]
    # Build the custom‐word button
    custom_button = InlineKeyboardButton('Enter custom word', callback_data='custom')
    keyboard = [
        suggestion_buttons[:3],
        suggestion_buttons[3:],    
        [custom_button],           
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(
        chat_id,
        'Choose your next guess:',
        reply_markup=reply_markup
    )

async def handle_guess(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button presses for word selection or custom entry."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith('guess:'):
        word = data.split(':', 1)[1]
        cands = context.chat_data.get('candidates', [])

        # if there's exactly one candidate left and it's this word, we're done
        if len(cands) == 1 and cands[0] == word:
            await query.message.reply_text(
                f"🎉 {word.upper()} should be the correct answer! Congratulations! 🎉\n"
                "Use /new to start a new game."
            )
            # (optionally) clear out game state here if you want:
            # context.chat_data.clear()
            return

        # otherwise, proceed as normal
        context.chat_data['current_guess'] = word
        await ask_feedback(query, context)
    elif data == 'custom':
        context.chat_data['awaiting_custom'] = True
        await query.message.reply_text('Please send your custom 5-letter guess.')

async def ask_feedback(source, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Prompt user to send feedback pattern for the current guess."""
    if isinstance(source, Update):
        chat_id = source.effective_chat.id
    else:
        chat_id = source.message.chat.id

    word = context.chat_data['current_guess']
    context.chat_data['awaiting_feedback'] = True
    await context.bot.send_message(
        chat_id,
        f"Send feedback for '{word.upper()}'.\n"
        f"\nUse '{GREEN}' for correct (🟩), '{YELLOW}' for wrong position (🟨), "
        f"and '{BLACK}' for absent (⬛)."
    )

def simulate_feedback(answer: str, guess: str) -> str:
    """Simulate Wordle feedback as a string of 'g', 'y', 'b'."""
    feedback = [''] * 5
    answer_chars = list(answer)

    # First pass for greens
    for i in range(5):
        if guess[i] == answer[i]:
            feedback[i] = GREEN
            answer_chars[i] = None  # consume

    # Second pass for yellows and blacks
    for i in range(5):
        if feedback[i]:
            continue
        if guess[i] in answer_chars:
            feedback[i] = YELLOW
            answer_chars[answer_chars.index(guess[i])] = None
        else:
            feedback[i] = BLACK
    return ''.join(feedback)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """General text handler for custom guesses or feedback patterns."""
    text = update.message.text.strip().lower()
    
    # Custom guess flow
    if context.chat_data.get('awaiting_custom'):
        if len(text) != 5 or text not in ALL_VALID_WORDS:
            await update.message.reply_text('Invalid word. Please send a valid 5-letter English word.')
            return
        context.chat_data['current_guess'] = text
        context.chat_data['awaiting_custom'] = False
        await ask_feedback(update, context)
        return

    # Feedback flow
    if context.chat_data.get('awaiting_feedback'):
        valid_chars = {GREEN, YELLOW, BLACK}
        if len(text) != 5 or any(ch not in valid_chars for ch in text):
            await update.message.reply_text(
                f"Invalid feedback. Use exactly 5 symbols from '{GREEN}', '{YELLOW}', '{BLACK}'."
            )
            return

        guess = context.chat_data['current_guess']
        pattern = text
        context.chat_data['awaiting_feedback'] = False

        # Check if all letters are correct (ggggg)
        if pattern == GREEN * 5:
            await update.message.reply_text(
                f"🎉 Congratulations! 🎉 \nYou got the correct answer: {guess.upper()}! \n"
                "Use /new to start a new game."
            )
            return

        # Record
        context.chat_data.setdefault('guesses', []).append(guess)
        context.chat_data.setdefault('patterns', []).append(pattern)

        # Filter candidates
        new_cands = [
            w for w in context.chat_data['candidates']
            if simulate_feedback(w, guess) == pattern
        ]
        context.chat_data['candidates'] = new_cands

        # Next suggestions
        await suggest_next(update, context)
        return

    # Fallback
    await update.message.reply_text(
        'Please use /new to start a game or select one of the provided options.'
    )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors caused by Updates."""
    logger.error("Exception while handling an update:", exc_info=context.error)

def main() -> None:
    """Start the bot."""
    load_dotenv()
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_TOKEN is not set in .env")

    app = ApplicationBuilder().token(token).build()

    # Command handlers
    app.add_handler(CommandHandler(['start', 'new'], start))
    # Inline button callbacks
    app.add_handler(CallbackQueryHandler(handle_guess))
    # Text messages (custom words & feedback)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    # Error handler
    app.add_error_handler(error_handler)

    logger.info('Starting Wordle Helper Bot...')
    app.run_polling()

if __name__ == '__main__':
    main()