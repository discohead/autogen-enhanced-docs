# Setting Up Your AutoGen Enhanced Documentation Repository

## Step 1: Create Repository on GitHub
1. Go to https://github.com/new
2. Create a new repository called `autogen-enhanced-docs` (or your preferred name)
3. **Important**: Do NOT initialize with README, .gitignore, or license
4. Keep it public or private as you prefer

## Step 2: Configure Git Remotes
After creating the repository on GitHub, run these commands:

```bash
# Rename the current origin to upstream (Microsoft's repo)
git remote rename origin upstream

# Add your GitHub repo as the new origin
git remote add origin https://github.com/discohead/autogen-enhanced-docs.git

# Verify the remotes are configured correctly
git remote -v
```

## Step 3: Push to Your Repository
```bash
# Push all branches to your repo
git push -u origin main

# Push all tags
git push origin --tags
```

## Step 4: Keeping Your Repo Updated with Microsoft's Changes

### Fetch latest changes from Microsoft
```bash
# Fetch from upstream
git fetch upstream

# Merge upstream changes into your main branch
git checkout main
git merge upstream/main

# Push updates to your repo
git push origin main
```

### Alternative: Rebase Strategy (cleaner history)
```bash
git fetch upstream
git checkout main
git rebase upstream/main
git push origin main --force-with-lease
```

## Important Notes
- Your CLAUDE.md files and custom documentation will be preserved
- Always commit your documentation changes before syncing with upstream
- Consider using a separate branch for your documentation work
- Resolve any merge conflicts carefully, prioritizing your custom docs

## Recommended Workflow
1. Make documentation changes in a feature branch
2. Regularly sync main with upstream
3. Rebase your feature branch on updated main
4. Merge your documentation changes back to main