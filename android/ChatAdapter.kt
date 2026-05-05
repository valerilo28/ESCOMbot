package com.example.escombot

import android.animation.ObjectAnimator
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import io.noties.markwon.Markwon

class ChatAdapter(private val messageList: List<Message>) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private val TYPE_USER = 1
    private val TYPE_BOT = 2
    private val TYPE_TYPING = 3

    override fun getItemViewType(position: Int): Int = when {
        messageList[position].isTyping -> TYPE_TYPING
        messageList[position].isBot -> TYPE_BOT
        else -> TYPE_USER
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return when (viewType) {
            TYPE_TYPING -> TypingViewHolder(
                inflater.inflate(R.layout.item_message_typing, parent, false)
            )
            TYPE_BOT -> BotViewHolder(
                inflater.inflate(R.layout.item_message_bot, parent, false)
            )
            else -> UserViewHolder(
                inflater.inflate(R.layout.item_message_user, parent, false)
            )
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        when (holder) {
            is TypingViewHolder -> holder.bind()
            is BotViewHolder -> holder.bind(messageList[position])
            is UserViewHolder -> holder.bind(messageList[position])
        }
    }

    override fun getItemCount() = messageList.size

    override fun onViewAttachedToWindow(holder: RecyclerView.ViewHolder) {
        super.onViewAttachedToWindow(holder)
        if (holder !is TypingViewHolder && holder.adapterPosition == messageList.size - 1) {
            holder.itemView.startAnimation(
                android.view.animation.AnimationUtils.loadAnimation(
                    holder.itemView.context, R.anim.fade_in
                )
            )
        }
    }

    // --- ViewHolders ---

    class TypingViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        fun bind() {
            listOf(R.id.dot1, R.id.dot2, R.id.dot3).forEachIndexed { i, id ->
                val dot = itemView.findViewById<View>(id)
                ObjectAnimator.ofFloat(dot, "translationY", 0f, -10f, 0f).apply {
                    duration = 500
                    startDelay = (i * 160).toLong()
                    repeatCount = ObjectAnimator.INFINITE
                    start()
                }
            }
        }
    }

    class BotViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val markwon = Markwon.create(itemView.context)
        fun bind(message: Message) {
            val textView = itemView.findViewById<TextView>(R.id.textMessageBot)
            textView.setTextIsSelectable(true)
            markwon.setMarkdown(textView, message.text)
        }
    }

    class UserViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        fun bind(message: Message) {
            itemView.findViewById<TextView>(R.id.textMessageUser).text = message.text
        }
    }
}