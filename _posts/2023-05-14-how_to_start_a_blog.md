# How to start a blog

So you want to start a blog. Or perhaps that's bold of me to assume straight up, so let's take a step back. If you haven't been living under a rock, you probably have a fair idea of what a blog is. But just to make sure we're starting on the same page, in summary a blog is a website presenting informative posts written in a conversation style. Actually, the word "blog" is an abbreviation of "weblog". In the early days of the internet, when everything was novel and interesting, people would blog about anything and everything. Nowadays, blogs have evolved into specific and informative posts, generally about a distinct subject matter.

But why start a blog? When I consider what exactly I could blog about - what I'm qualified to share, perhaps - I relate to a piece Rachael Thomas, a co-founder of [fast.ai](https://www.fast.ai/), wrote in her own [blog post](https://medium.com/@racheltho/why-you-yes-you-should-blog-7d2544ac1045):

> My partner, Jeremy, had been suggesting for years that I should start blogging, and I’d respond “I don’t have anything to say.” This wasn’t true. What I meant was that I didn’t feel confident, and I felt like the things I could write had already been written about by people with more expertise or better writing skills than me.

Fortunately, for all of us in this position, Rachael goes on to assure us that blog posts don't have to be novel or groundbreaking to be interesting and informative. Instead, one potential use of a blog is to document your personal learning experiences. This is exactly what I'll be aiming for as I kick of this blog with my first post.

Hopefully, I'll discover that starting a blog can indeed be a rewarding experience, as many will have it (and not least because there are marks at stake for a university course hinging on my starting of this blog).

## Creating a blog using `fast_template`

This blog was created using `fast_template`, courtesy of fast.ai. It's free and based on powerful and flexible foundations like [git](https://git-scm.com/) and [Jekyll](https://jekyllrb.com/). The blog itself is hosted using [GitHub Pages](https://pages.github.com/) (hence the `.github.io` URL which you may have noticed). To setup the blog, I followed the [very simple guide](https://www.fast.ai/posts/2020-01-16-fast_template.html) provided by Jemery Howard, another co-founder of fast.ai.

First up, a GitHub account is necessary for `fast_template` to setup and host the blog using GitHub pages. With this in hand, click the link: https://github.com/fastai/fast_template/generate. This will prompt you to create a new repository for your blog. It should look something like this:

![](/images/2023-05-14_fast_template_step_1.png)

It's important that the repository name is **exactly** your GitHub username followed by `.github.io`. Since I've already created my blog, my particular repository name is already taken.

Once you create the repository, the initial contents are pretty self-explanatory and were enough to get me started. Otherwise, Jeremy's post continues to provide detailed advice on how to create a your first post and some tips on writing content using the Markdown language.

At any point after creating your repository, you can view your blog at the url of your repository name. If you've just committed changes to your blog, it may take a couple of minutes for these to become visible, as GitHub needs to re-render the parts that have changed, then re-deploy the website.

## Writing quality content for your new blog

Now that you have a blog platform with a bunch of default content, what do you write for your first post? And your next post? And every subsequent post?

I guess the unsatisfyingly unspecific answer is: whatever you want. For me, I've gone with Rachael's suggestion:

> You are best positioned to help people one step behind you. The material is still fresh in your mind. Many experts have forgotten what it was like to be a beginner (or an intermediate) and have forgotten why the topic is hard to understand when you first hear it. The context of your particular background, your particular style, and your knowledge level will give a different twist to what you’re writing about.

That is, I'll be writing about recent discoveries and learnings; what would've helped me as I started doing the thing I just finished.

For guidance on formatting and writing style, I picked out three blog posts that I enjoyed reading and will briefly dissect them to see what tips and tricks I can adopt. These are:

- [*Advice for better blog posts*](https://www.fast.ai/posts/2019-05-13-blogging-advice.html) by Rachael Thomas;

- [*Introducing ChatGPT*](https://openai.com/blog/chatgpt) by OpenAI; and

- [*Sick Tricks and Sticky Grips*](https://www.bostondynamics.com/resources/blog/sick-tricks-and-tricky-grips) by Boston Dynamics.

### *Advice for better blog posts* by Rachael Thomas

Given the subject of this post, you'd naturally expect a lot of great guidance here. I took away the following tips in particular:

- **Strong opening sentences.** The first few sentences make a strong impact and give you reason to be interested in the content that follows.
- **Links to context and additional information.** Plenty of links are provided so the reader can go through contextual or additional information as they desire, which keeps the content focused.
- **Information chunked for better clarity.** The content is divided into sections, paragraphs and points in a way that doesn't overwhelm the reader.
- **Markdown features for emphasis.** Markdown enables text formatting features like bold, italics and quotation blocks. These are used - but not overused - to emphasise important points.

### *Introducing ChatGPT* by OpenAI

I've always enjoyed reading posts By OpenAI. In fact, OpenAI's posts on [OpenAI Five](https://openai.com/research/openai-five) and [GPT-2](https://openai.com/research/gpt-2-1-5b-release) are some of what got me into machine learning in the first place. In their post introducing ChatGPT, I took note of:

- **Complex topics are simply explained.** OpenAI somehow manage to distill cutting edge research into simple and easy-to-understand blog posts. Explaining complex topics in a way that can be understood by non-experts is a powerful skill.

- **Sample results.** Sample results are provided early in the post, which demonstrate the impressiveness of ChatGPT (that we're now all familiar with). This piques your interest to continue reading and find out how they did it.

- **Attractive diagrams.** Deep learning networks are a complex thing, but can be made a bit more intuitive by well-designed diagrams. As the cliche goes, a picture is worth a thousand words?

- **Clearly defined sections.** The post is divided into very clear sections, sort of like a research paper. While this defies the conversational tone somewhat, I think it's worthwhile for the clarity it affords in highly scientific contexts.

### *Sick Tricks and Sticky Grips* by Boston Dynamics

Watching the Boston Dynamics Atlas do backflips and parkour is non-stop fun. In this post, Boston Dynamics discuss the addition of hands for Atlas. I liked the following in particular:

- **Quirky title.** The title is fun and matches the general tone of Atlas demonstrations. Personally, this makes it more attractive to read when I'm not after anything in particular.

- **Lead with an image.** The post opens with an appropriate image of Atlas demonstrating the developments discussed in the post. This makes the post more visually attractive. It's also easier to follow the content when you can see what is being talked about.

- **Interview style.** The post is presented in the style of an interview with a lead engineer. This is pretty common, and I enjoy the sense of hearing a personal update from a real expert.

## In summary

Now, none of the listed post examples actually have a proper conclusion, but I feel like it's appropriate to sign off. In this post - my first post - I've touched on why you might want to start a blog, how this blog was created, and things I like about a few great posts from other blogs. Hopefully, the last one starts to show in my own posts too. Perhaps it's time to start your own blog?
